"""Generate a step's reply until it is usable, sending problems back to the LLM.

A reply is rejected when it fails the step's own checks or, for steps with a
check prompt, when the LLM finds in a self-check that the result is not
consistent. Each rejection goes back to the generating conversation as
feedback; a step that has no usable reply after max_generations is dropped.
"""

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from awm.core.augment.draft import LLMStep, Rejection
from awm.tools import tools_robust_json_loads

logger = logging.getLogger(__name__)

# sends a batch of conversations and returns one reply per conversation (None if failed)
Complete = Callable[[list[list[dict]]], Sequence[str | None]]

# self-check replies that are not a valid verdict are asked again this many times
MAX_CHECK_RETRY = 2

INVALID_JSON = Rejection("Reply with a single valid JSON object and nothing else.")


class SelfCheckReply(BaseModel):
    consistent: bool
    problems: list[str] = []


@dataclass
class _Attempt[T]:
    step: LLMStep[T]
    # the generating conversation, with earlier replies and their feedback
    messages: list[dict]
    generations: int = 0
    # a result the step accepted, waiting for its self-check
    candidate: T | None = None
    candidate_reply: str = ""
    check_failures: int = 0
    history: list[str] = field(default_factory=list)

    @property
    def checking(self) -> bool:
        return self.candidate is not None


def _bullets(problems: list[str]) -> str:
    return "\n".join(f"- {p}" for p in problems)


def _send_back(
    attempt: _Attempt, reply: str | None, problems: list[str], max_generations: int
) -> bool:
    """Queue another generation with the problems as feedback; False when out of tries."""
    attempt.history.extend(problems)
    if attempt.generations >= max_generations:
        return False
    if reply:
        attempt.messages = [
            *attempt.messages,
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": attempt.step.feedback_prompt.format(
                    problems=_bullets(problems)
                ),
            },
        ]
    return True


def _finish[T](step: LLMStep[T], reply: str | None) -> T | Rejection:
    parsed = tools_robust_json_loads(reply) if reply else None
    return step.finish(parsed) if isinstance(parsed, dict) else INVALID_JSON


def _on_generation[T](
    attempt: _Attempt[T],
    reply: str | None,
    result: T | Rejection,
    max_generations: int,
) -> T | bool:
    attempt.generations += 1
    if isinstance(result, Rejection):
        return _send_back(attempt, reply, [result.reason], max_generations)
    if attempt.step.check_prompt is None:
        return result
    attempt.candidate = result
    attempt.candidate_reply = reply or ""
    attempt.check_failures = 0
    return True


def _on_self_check[T](
    attempt: _Attempt[T], reply: str | None, max_generations: int
) -> T | bool:
    try:
        verdict = SelfCheckReply.model_validate(
            tools_robust_json_loads(reply) if reply else None
        )
    except ValidationError:
        attempt.check_failures += 1
        return attempt.check_failures <= MAX_CHECK_RETRY
    candidate = attempt.candidate
    assert candidate is not None
    if verdict.consistent:
        return candidate
    attempt.candidate = None
    problems = verdict.problems or ["The result is not consistent."]
    return _send_back(attempt, attempt.candidate_reply, problems, max_generations)


def _request(attempt: _Attempt) -> list[dict]:
    if not attempt.checking:
        return attempt.messages
    check_prompt = attempt.step.check_prompt
    assert check_prompt is not None
    return [
        *attempt.messages,
        {"role": "assistant", "content": attempt.candidate_reply},
        {"role": "user", "content": check_prompt(attempt.candidate)},
    ]


def refine[T](
    steps: Sequence[LLMStep[T]],
    complete: Complete,
    max_generations: int,
    workers: int = 1,
) -> list[T]:
    """Results of the steps that got a usable reply within max_generations.

    workers: replies are finished in parallel, for steps whose finish is slow
    (it starts servers).
    """
    pending = [_Attempt(step, list(step.messages)) for step in steps]
    results: list[T] = []
    dropped: list[_Attempt[T]] = []
    while pending:
        replies = list(complete([_request(a) for a in pending]))
        to_finish = [
            (a, r) for a, r in zip(pending, replies, strict=True) if not a.checking
        ]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            finished = list(pool.map(lambda ar: _finish(ar[0].step, ar[1]), to_finish))
        by_attempt = {id(a): result for (a, _), result in zip(to_finish, finished)}

        still_pending = []
        for attempt, reply in zip(pending, replies, strict=True):
            if attempt.checking:
                outcome = _on_self_check(attempt, reply, max_generations)
            else:
                outcome = _on_generation(
                    attempt, reply, by_attempt[id(attempt)], max_generations
                )
            if outcome is True:
                still_pending.append(attempt)
            elif outcome is False:
                dropped.append(attempt)
            else:
                results.append(outcome)
        pending = still_pending
    if dropped:
        logger.warning(
            f"Dropped {len(dropped)} LLM steps with no usable reply in {max_generations} generations"
        )
        for attempt in dropped:
            logger.debug(f"{attempt.step.label}: {attempt.history}")
    return results
