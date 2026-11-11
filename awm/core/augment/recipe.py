"""Recipes: augmentation parts applied one after another to source tasks.

A recipe is written as part names joined by "+", e.g.
"tool_rename+task_ambiguous+miss_param"; a single part is a recipe of one.
Each task a recipe makes is an environment of its own,
<scenario>__<recipe>__<source task index>, so that tasks augmented differently
never share a server. Its server code presents the tools as the task's view
says (bake_view), so any MCP client, an RL trainer included, sees the
augmented tools without reading the task meta.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from awm.core.augment.dataset import EnvVariant, env_scenario
from awm.core.augment.draft import Draft, finalize, properties
from awm.core.augment.env_parts import ParamRename, ToolRename, ToolRenameRandom
from awm.core.augment.followup_parts import MissFunc, MissParam
from awm.core.augment.part import Part, PartContext
from awm.core.augment.replay import replay, source_call_errors, tool_set_problems
from awm.core.augment.task_parts import (
    TaskAmbiguous,
    TaskDistractorNames,
    TaskNoToolNames,
    TaskValueFormat,
)
from awm.core.augment.tool_parts import (
    AddParams,
    DistractorTools,
    ParamEnum,
    ParamFormat,
    ToolDocParaphrase,
)
from awm.core.server import ServerStartError
from awm.core.tool_view import ToolView, bake_view
from awm.core.view_server import VALUE_FORMATS
from awm.tools import normalize_scenario_name

logger = logging.getLogger(__name__)

RECIPE_SEPARATOR = "+"

PARTS: dict[str, Part] = {
    part.name: part
    for part in [
        AddParams(),
        DistractorTools(),
        ParamEnum(),
        ParamFormat(),
        ToolDocParaphrase(),
        TaskAmbiguous(),
        TaskNoToolNames(),
        TaskValueFormat(),
        TaskDistractorNames(),
        MissFunc(),
        MissParam(),
        ToolRename(),
        ToolRenameRandom(),
        ParamRename(),
    ]
}


@dataclass(frozen=True)
class Recipe:
    parts: tuple[Part, ...]

    @property
    def name(self) -> str:
        return RECIPE_SEPARATOR.join(p.name for p in self.parts)


def recipe_problems(parts: list[Part]) -> list[str]:
    """Why the parts cannot be applied in this order."""
    problems = []
    names = [p.name for p in parts]
    if dup := sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"{dup} appear more than once")
    first_task_part = next(
        (i for i, p in enumerate(parts) if not p.rebuilds_env), len(parts)
    )
    if late := [p.name for p in parts[first_task_part:] if p.rebuilds_env]:
        problems.append(
            f"{late} rebuild the environment and must come before {parts[first_task_part].name}"
        )
    for i, part in enumerate(parts):
        for other in parts[i + 1 :]:
            if both := sorted(part.writes & other.writes):
                problems.append(f"{part.name} and {other.name} both set {both}")
    return problems


def parse_recipe(text: str) -> Recipe:
    """The recipe of part names joined by "+"; raises ValueError if it cannot run."""
    names = [n.strip() for n in text.split(RECIPE_SEPARATOR)]
    if unknown := [n for n in names if n not in PARTS]:
        raise ValueError(
            f"unknown parts {unknown} in {text!r}, choose from {sorted(PARTS)}"
        )
    parts = [PARTS[n] for n in names]
    if problems := recipe_problems(parts):
        raise ValueError(f"recipe {text!r}: " + "; ".join(problems))
    return Recipe(tuple(parts))


def run_recipe(
    recipe: Recipe,
    drafts_by_scenario: dict[str, list[Draft]],
    ctx: PartContext,
    oversample: int = 2,
) -> list[EnvVariant]:
    """An environment per augmented task, up to num_per_part per source scenario.

    drafts_by_scenario: source drafts by normalized scenario name.
    """
    drafts = source_drafts(recipe, drafts_by_scenario, ctx, oversample)
    tried = len(drafts)
    for part in recipe.parts:
        drafts = part.apply(drafts, ctx)

    by_scenario: dict[str, list[Draft]] = {}
    for draft in sorted(
        drafts, key=lambda d: (normalize_scenario_name(d.scenario), d.source_task_idx)
    ):
        by_scenario.setdefault(normalize_scenario_name(draft.scenario), []).append(
            draft
        )

    def scenario_envs(members: list[Draft]) -> list[EnvVariant]:
        return served_envs(recipe, members, ctx)

    with ThreadPoolExecutor(max_workers=max(1, ctx.workers)) as pool:
        envs = [
            e for found in pool.map(scenario_envs, by_scenario.values()) for e in found
        ]
    logger.info(f"Recipe {recipe.name}: {len(envs)} tasks from {tried} tried")
    return envs


def source_drafts(
    recipe: Recipe,
    drafts_by_scenario: dict[str, list[Draft]],
    ctx: PartContext,
    oversample: int,
) -> list[Draft]:
    """Source drafts the recipe can start on, in a seeded order per scenario.

    oversample times num_per_part are taken per scenario, since a part can
    reject a task (nothing to rename, a rewrite that failed validation, ...)
    and a server can fail its check.
    """
    first = recipe.parts[0]
    drafts: list[Draft] = []
    for scenario, scenario_drafts in drafts_by_scenario.items():
        candidates = [d for d in scenario_drafts if first.applicable(d)]
        ctx.rng(recipe.name, scenario).shuffle(candidates)
        drafts.extend(candidates[: ctx.num_per_part * oversample])
    return drafts


def build_env(recipe: Recipe, draft: Draft, ctx: PartContext) -> EnvVariant | None:
    """The environment of the draft's task; None without server code to serve its view."""
    code = draft.code
    if draft.has_view:
        if code is None:
            code = ctx.env_code.get(normalize_scenario_name(draft.scenario))
        if code is None:
            logger.warning(
                f"{draft.scenario}: no server code to serve the view of {recipe.name}"
            )
            return None
        code = bake_view(code, draft.view)
    return EnvVariant(
        scenario=env_scenario(draft.scenario, recipe.name, draft.source_task_idx),
        source_scenario=draft.scenario,
        augmentation=recipe.name,
        code=code,
        variants=[finalize(draft)],
    )


def served_envs(
    recipe: Recipe, drafts: list[Draft], ctx: PartContext
) -> list[EnvVariant]:
    """Environments of a scenario's drafts, in order, up to num_per_part that serve their task."""
    envs: list[EnvVariant] = []
    for draft in drafts:
        if len(envs) == ctx.num_per_part:
            break
        env = build_env(recipe, draft, ctx)
        if env is None:
            continue
        if problems := view_problems(ctx, env, draft):
            logger.warning(f"{env.scenario} dropped: {'; '.join(problems)}")
            continue
        envs.append(env)
    return envs


# ------------------------------------------------------- the served view


def presented_calls(calls: list[dict], view: ToolView) -> list[dict]:
    """The calls as an agent makes them on the tools the view presents."""
    out = []
    for call in calls:
        arguments = dict(call.get("arguments") or {})
        for param, fmt in view.param_formats.get(call["tool_name"], {}).items():
            if arguments.get(param) is not None:
                arguments[param] = VALUE_FORMATS[fmt].to_agent(arguments[param])
        out.append({**call, "arguments": arguments})
    return out


def view_listing_problems(listed: list[dict], draft: Draft) -> list[str]:
    """What of the draft's view the listed tools do not show."""
    view = draft.view
    tools = {t["name"]: t for t in listed}
    expected = set(draft.tool_names) | {t["name"] for t in view.distractor_tools}
    problems = tool_set_problems(set(tools), expected)
    for presented in (
        view.param_descriptions,
        view.extra_params,
        view.param_enums,
        view.param_formats,
    ):
        for name, params in presented.items():
            missing = sorted(set(params) - set(properties(tools.get(name, {}))))
            if name in tools and missing:
                problems.append(f"{name} does not list the parameters {missing}")
    reworded = sorted(
        name
        for name, text in view.tool_descriptions.items()
        if name in tools and tools[name].get("description") != text
    )
    if reworded:
        problems.append(f"{reworded} are not listed with their new descriptions")
    return problems


def solved_on_source(ctx: PartContext, draft: Draft) -> bool:
    """Whether the draft's reference calls solve it on its server without the view."""
    if draft.call_errors is not None:
        # replayed on the server an earlier part rebuilt
        return True
    code = ctx.env_code[normalize_scenario_name(draft.scenario)]
    return source_call_errors(ctx.replay, code, draft) is not None


def view_problems(ctx: PartContext, env: EnvVariant, draft: Draft) -> list[str]:
    """Why the environment's server does not serve the task's view.

    The server has to list the tools the view presents, and the task's
    reference calls, made on the presented tools, have to solve it if they
    solve it without the view. Without a view there is nothing to check.
    """
    if not draft.has_view:
        return []
    assert env.code is not None
    calls = presented_calls(draft.reference_calls or [], draft.view)
    try:
        result = replay(
            ctx.replay,
            env.code,
            env.source_scenario,
            calls,
            draft.source_verifier_code,
            draft.reference_answer,
        )
    except ServerStartError as e:
        return [f"the server does not start: {e}"]
    problems = view_listing_problems(result.tools, draft)
    # a run can depend on what a replay does not reproduce, e.g. the time
    if calls and result.reward != "complete" and solved_on_source(ctx, draft):
        problems.append(f"the presented reference calls are graded {result.reward}")
    return problems
