"""The interface every augmentation part implements.

A part takes a batch of drafts and returns the drafts it changed; drafts it
cannot handle are dropped. Parts of a recipe (recipe.py) are applied one after
another, each to the drafts the previous one returned, so every part sees what
the earlier ones did.

- TaskPart changes each draft on its own: how the tools are presented, the task
  text, a follow-up turn.
- EnvPart (env_parts.py) rewrites a scenario's server code once for all its
  drafts.
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from awm.core.augment.draft import Draft, LLMStep
from awm.core.augment.replay import ReplayContext
from awm.tools import normalize_scenario_name


class RunLLM(Protocol):
    """Runs a batch of LLM steps and returns the results of the usable replies.

    workers: replies finished in parallel, for steps whose finish is slow.
    """

    def __call__[T](self, steps: Sequence[LLMStep[T]], workers: int = 1) -> list[T]: ...


@dataclass
class PartContext:
    seed: int
    # augmented tasks per scenario and recipe
    num_per_part: int
    run_llm: RunLLM
    replay: ReplayContext
    # normalized scenario -> server code of the source scenario
    env_code: dict[str, str] = field(default_factory=dict)
    # concurrent replays of rewritten server code
    workers: int = 1

    def rng(self, *keys: object) -> random.Random:
        """A generator seeded with the run's seed and the keys."""
        return random.Random(":".join(str(k) for k in (self.seed, *keys)))


class Part(ABC):
    """One augmentation; subclasses set name and implement apply."""

    name: ClassVar[str]
    # what the part sets on a draft that another part of a recipe must not set
    # too, e.g. a ToolView field or the follow-up turn
    writes: ClassVar[frozenset[str]] = frozenset()
    # parts that rewrite the server code come before the others in a recipe,
    # since tool views and task texts refer to the tool names
    rebuilds_env: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    needs_tools: ClassVar[bool] = False

    def applicable(self, draft: Draft) -> bool:
        if self.needs_reference and not draft.reference_calls:
            return False
        return not (self.needs_tools and not draft.tools)

    @abstractmethod
    def apply(self, drafts: list[Draft], ctx: PartContext) -> list[Draft]:
        """The drafts changed by the part; LLM steps are sent as one batch."""


class TaskPart(Part):
    """A part that changes each draft on its own; subclasses implement apply_one."""

    @abstractmethod
    def apply_one(
        self, draft: Draft, rng: random.Random
    ) -> Draft | LLMStep[Draft] | None:
        """The changed draft, a step that produces it, or None if it cannot change."""

    def apply(self, drafts: list[Draft], ctx: PartContext) -> list[Draft]:
        produced: list[Draft] = []
        steps: list[LLMStep[Draft]] = []
        for draft in drafts:
            if not self.applicable(draft):
                continue
            rng = ctx.rng(
                self.name,
                normalize_scenario_name(draft.scenario),
                draft.source_task_idx,
            )
            out = self.apply_one(draft, rng)
            if isinstance(out, LLMStep):
                steps.append(out)
            elif out is not None:
                produced.append(out)
        if steps:
            produced.extend(ctx.run_llm(steps))
        return produced
