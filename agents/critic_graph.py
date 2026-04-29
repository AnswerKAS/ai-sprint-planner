import asyncio
import logging
import re
from typing import Any, TypedDict

import redis as redis_lib
from langchain.agents import create_agent
from langchain_community.storage import RedisStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from agents.logging_utils import log_agent_messages
from agents.tools.common_tools import TEAM_CAPACITY
from agents.tools.critic_tools import make_critic_sp_tool
from store.redis import save_critic_consultation_to_redis, save_critic_iteration_to_redis
from tasks.model import SprintTask

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
MAX_RELAX_LOWER_AFTER = 7

_CONSULT_SYSTEM = (
    "/no_think\n\n"
    "Ты эксперт по планированию спринта. "
    "Давай краткие и чёткие рекомендации — только task_id и причину."
)


class CriticState(TypedDict):
    iteration: int
    plan: str
    selected_task_ids: list[str]  # authoritative deduplicated list from tool call
    feedback: str
    validated: bool
    consultation: str  # collected advice from specialist agents
    prev_remove_recs: list[str]   # task IDs already recommended for removal (skip in next round)
    prev_add_recs: list[str]      # task IDs already recommended for addition (skip in next round)


def _parse_task_ids(text: str) -> list[str]:
    return re.findall(r"\b[A-Z]+-\d+\b", text)


def _fmt_task(task: SprintTask) -> str:
    s = f"- {task.task_id}: {task.title} [SP: {task.sp} | Приоритет: {task.priority}"
    if task.escalation_count:
        s += f" | Эскалации: {task.escalation_count}"
    if task.rice is not None:
        s += f" | RICE: {task.rice}"
    if task.stage:
        s += f" | Стадия: {task.stage}"
    return s + "]"


def _find_sp_tool_result(messages: list[Any]) -> str | None:
    for msg in messages:
        if type(msg).__name__ == "ToolMessage":
            if getattr(msg, "name", "") == "check_sprint_sp":
                return getattr(msg, "content", None)
    return None


def _extract_task_ids_from_tool_call(messages: list[Any]) -> list[str] | None:
    """Extract task_ids from the check_sprint_sp tool call arguments — more reliable than regex parsing."""
    for msg in messages:
        if type(msg).__name__ == "AIMessage":
            for tc in getattr(msg, "tool_calls", []):
                if tc.get("name") == "check_sprint_sp":
                    ids = tc.get("args", {}).get("task_ids", [])
                    if ids:
                        return list(dict.fromkeys(ids))  # deduplicate, preserve order
    return None


async def _consult_specialist(
    llm: ChatOllama,
    specialist_name: str,
    tasks_selected: list[SprintTask],
    tasks_available: list[SprintTask],
    sp_status: str,
    sp_gap: float,
) -> str | None:
    """Async LLM call to ask a specialist what to remove (over) or add (under)."""
    if sp_status == "over":
        if tasks_selected:
            tasks_str = "\n".join(_fmt_task(t) for t in tasks_selected)
            prompt = (
                f"Ты — {specialist_name}.\n"
                f"В текущем спринте выбраны следующие твои задачи:\n{tasks_str}\n\n"
                f"Спринт превышает допустимую ёмкость на {sp_gap:.1f} SP. "
                f"Какие из твоих задач можно убрать без критической потери бизнес-ценности?\n"
                f"Квотные задачи (quota > 0) убирать нельзя ни при каких условиях.\n"
                f"Формат — одна задача на строку: <task_id>: <причина>"
            )
        elif tasks_available:
            avail_str = "\n".join(_fmt_task(t) for t in tasks_available[:10])
            prompt = (
                f"Ты — {specialist_name}.\n"
                f"Твои задачи пока не включены в спринт, но план превышает ёмкость на {sp_gap:.1f} SP.\n"
                f"Следующие твои задачи могут стать заменой более дорогим задачам:\n{avail_str}\n\n"
                f"Если среди них есть задачи, которые стоит предложить как более дешёвую альтернативу "
                f"дорогим задачам из других категорий — назови их.\n"
                f"Формат — одна задача на строку: <task_id>: <причина>"
            )
        else:
            return None
    else:
        if not tasks_available:
            return None
        tasks_str = "\n".join(_fmt_task(t) for t in tasks_available)
        prompt = (
            f"Ты — {specialist_name}.\n"
            f"В спринт можно добавить ещё {sp_gap:.1f} SP.\n"
            f"Следующие твои задачи ещё не включены в план:\n{tasks_str}\n\n"
            f"Какие из них рекомендуешь добавить?\n"
            f"Формат — одна задача на строку: <task_id>: <причина>"
        )

    response = await llm.ainvoke([
        SystemMessage(content=_CONSULT_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = response.content if isinstance(response.content, str) else str(response.content)
    return text.strip() or None


def build_critic_graph(
    config: dict,
    task_list: list[SprintTask],
    store: RedisStore,
    redis_client: redis_lib.Redis,
    session_id: str,
    team_name: str,
    candidates_context: str = "",
):
    sys_prompt = "/no_think\n\n" + config["system_prompt"] + "\n\n" + config["limitations"]
    model_name = config.get("model", "qwen3:4b-instruct")
    agent_name = config.get("name", "critic_agent")

    # candidates_context is injected directly into every prompt — no need for the get_all_candidates tool
    critic_agent = create_agent(
        model=ChatOllama(model=model_name, temperature=0, reasoning=False),
        tools=[make_critic_sp_tool(task_list, team_name)],
        system_prompt=sys_prompt,
        name=agent_name,
        debug=False,
        store=store,
    )
    consult_llm = ChatOllama(model=model_name, temperature=0, reasoning=False)

    task_index = {task.task_id: task for task in task_list}
    team_task_ids: set[str] = {t.task_id for t in task_list if t.team == team_name}
    quota_task_ids: frozenset[str] = frozenset(
        t.task_id for t in task_list if t.team == team_name and t.quota and t.quota > 0
    )
    has_quota_tasks: bool = bool(quota_task_ids)
    capacity = TEAM_CAPACITY.get(team_name, 0.0)
    lower = round(capacity * 0.9, 2)   # минимум: 90%
    upper = round(capacity * 1.1, 2)   # жёсткий лимит: 110%
    RELAX_LOWER_AFTER = MAX_RELAX_LOWER_AFTER

    _PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    _CATEGORY_ORDER = {"incident": 0, "project": 1, "task": 2}

    # Compile one pattern for all known task IDs — O(1) per extraction call instead of O(N)
    _id_pattern: re.Pattern | None = (
        re.compile(
            r'(?<!\w)(?:'
            + '|'.join(re.escape(t) for t in sorted(team_task_ids, key=len, reverse=True))
            + r')(?!\w)'
        )
        if team_task_ids else None
    )

    def _extract_known_ids(text: str) -> list[str]:
        if not _id_pattern:
            return []
        return list(dict.fromkeys(_id_pattern.findall(text)))

    def _greedy_fit(candidate_ids: list[str]) -> tuple[list[str], float]:
        """Trim candidate list to fit within upper SP cap.
        Quota tasks are always included. Non-quota tasks are added greedily:
        incident first, then project, then task; within each category by priority then escalations.
        Returns (selected_ids, total_sp).
        """
        candidates = [task_index[tid] for tid in candidate_ids if tid in task_index]
        quota = [t for t in candidates if t.quota and t.quota > 0]
        non_quota = sorted(
            [t for t in candidates if not (t.quota and t.quota > 0)],
            key=lambda t: (
                _CATEGORY_ORDER.get(t.category, 3),
                _PRIORITY_ORDER.get(t.priority, 4),
                -(t.escalation_count or 0),
            ),
        )

        selected = [t.task_id for t in quota]
        total = sum(task_index[tid].sp for tid in selected if tid in task_index)

        for t in non_quota:
            if total + t.sp <= upper:
                selected.append(t.task_id)
                total += t.sp

        return selected, round(total, 2)

    def _format_greedy_plan(task_ids: list[str], total_sp: float) -> str:
        lines = [
            f"{tid} [SP: {task_index[tid].sp} | Категория: {task_index[tid].category} | Приоритет: {task_index[tid].priority}]"
            for tid in task_ids
            if tid in task_index
        ]
        utilization = round(total_sp / capacity * 100, 1) if capacity else 0
        lines.append(f"\nИТОГО: {total_sp} SP | ЁМКОСТЬ: {capacity} SP | ЗАГРУЗКА: {utilization}%")
        return "\n".join(lines)

    def _split_tasks(task_ids_in_plan: set[str], category: str):
        selected, available = [], []
        for t in task_list:
            if t.team != team_name or t.category != category:
                continue
            (selected if t.task_id in task_ids_in_plan else available).append(t)
        return selected, available

    # ── nodes ──────────────────────────────────────────────────────────────

    async def critic_node(state: CriticState) -> dict:
        iteration = state["iteration"]
        feedback = state.get("feedback", "")
        previous_plan = state.get("plan", "")
        consultation = state.get("consultation", "")

        logger.info(
            "┌── %s | iteration %d/%d ───────────────────────────────",
            agent_name, iteration + 1, MAX_ITERATIONS,
        )

        sp_rule = (
            f"Ёмкость команды: {capacity} SP. "
            f"Идеальный диапазон: {lower}–{upper} SP (90–110%). "
            f"Жёсткий лимит: {upper} SP — превышать нельзя ни при каких условиях."
        )
        quota_note = (
            "Квотные задачи ОБЯЗАТЕЛЬНЫ — включи их все."
            if has_quota_tasks
            else "У этой команды нет квотных задач — это нормально, продолжай без них."
        )
        if iteration == 0:
            prompt = (
                f"Составь спринт для команды {team_name}.\n"
                f"{sp_rule}\n\n"
                f"Кандидаты от агентов-специалистов:\n{candidates_context}\n\n"
                f"Шаг 1: выбери задачи из списка выше. {quota_note}\n"
                f"Шаг 2: вызови check_sprint_sp(task_ids=[...]) "
                f"для проверки — убедись, что SP в диапазоне [{lower}, {upper}]."
            )
        else:
            prompt = (
                f"Итерация {iteration + 1}. Пересмотри план спринта для команды {team_name}.\n\n"
                f"Кандидаты от агентов-специалистов:\n{candidates_context}\n\n"
                f"Текущий план:\n{previous_plan}\n\n"
                f"Проблема валидатора: {feedback}\n\n"
                f"Рекомендации агентов-специалистов:\n{consultation}\n\n"
                f"{sp_rule}\n"
                f"Шаг 1: скорректируй план согласно рекомендациям. {quota_note}\n"
                f"Шаг 2: вызови check_sprint_sp(task_ids=[...]) и убедись, что SP в [{lower}, {upper}]."
            )

        result = await critic_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        messages = result.get("messages", [])
        log_agent_messages(messages, agent_name)

        sp_tool_result = _find_sp_tool_result(messages)
        if sp_tool_result:
            logger.info("│ [%s] SP tool result: %s", agent_name, sp_tool_result[:300])
        else:
            logger.warning(
                "│ [%s] Agent did NOT call check_sprint_sp — validator will check independently",
                agent_name,
            )

        last = messages[-1] if messages else None
        plan = getattr(last, "content", "") if last else ""
        if not isinstance(plan, str):
            plan = str(plan)

        # ── Шаг 1: авторитетный источник — аргументы tool-вызова check_sprint_sp
        tool_task_ids = _extract_task_ids_from_tool_call(messages)
        if tool_task_ids:
            task_ids = tool_task_ids
            logger.info("│ [%s] task_ids: from check_sprint_sp tool call (%d)", agent_name, len(task_ids))
        else:
            # ── Шаг 2: стандартный regex [A-Z]+-\d+ по тексту плана
            task_ids = list(dict.fromkeys(_parse_task_ids(plan)))
            if task_ids:
                logger.warning("│ [%s] task_ids: regex fallback (%d)", agent_name, len(task_ids))
            else:
                # ── Шаг 3: поиск по всем известным ID команды (включая нестандартные форматы)
                task_ids = _extract_known_ids(plan)
                if task_ids:
                    logger.warning("│ [%s] task_ids: known-id scan fallback (%d)", agent_name, len(task_ids))
                else:
                    # ── Шаг 4: LLM не вернул ничего — жадный отбор из всех кандидатов команды
                    logger.warning(
                        "│ [%s] task_ids: LLM вернул пустой план — применяем greedy по всем кандидатам",
                        agent_name,
                    )
                    task_ids, total_sp = _greedy_fit(list(team_task_ids))
                    plan = _format_greedy_plan(task_ids, total_sp)

        total_sp = round(sum(task_index[tid].sp for tid in task_ids if tid in task_index), 2)
        not_found = [tid for tid in task_ids if tid not in task_index]

        # Force-include quota tasks the LLM might have omitted
        if has_quota_tasks:
            missing_quota = quota_task_ids - set(task_ids)
            if missing_quota:
                logger.warning(
                    "│ [%s] Квотные задачи отсутствуют в плане: %s — добавляем принудительно",
                    agent_name, missing_quota,
                )
                task_ids = list(dict.fromkeys([*missing_quota, *task_ids]))
                total_sp = round(
                    sum(task_index[tid].sp for tid in task_ids if tid in task_index), 2
                )

        if total_sp > upper:
            logger.warning(
                "│ [%s] SP=%.2f превышает лимит %.2f — применяем жадный отбор",
                agent_name, total_sp, upper,
            )
            task_ids, total_sp = _greedy_fit(task_ids)
            plan = _format_greedy_plan(task_ids, total_sp)
            logger.info(
                "│ [%s] После жадного отбора: %d задач, SP=%.2f",
                agent_name, len(task_ids), total_sp,
            )

        logger.info(
            "│ [%s] Tasks: %d | SP: %.2f | target [%.2f, %.2f]",
            agent_name, len(task_ids), total_sp, lower, upper,
        )
        if not_found:
            logger.warning("│ [%s] Unknown task IDs: %s", agent_name, not_found)
        logger.info("└───────────────────────────────────────────────────────")

        return {
            "iteration": iteration + 1,
            "plan": plan,
            "selected_task_ids": task_ids,
            "feedback": state.get("feedback", ""),
            "validated": False,
            "consultation": "",
        }

    def validator_node(state: CriticState) -> dict:
        plan = state.get("plan", "")
        iteration = state["iteration"]

        task_ids = state.get("selected_task_ids") or list(dict.fromkeys(_parse_task_ids(plan)))
        total_sp = round(
            sum(task_index[tid].sp for tid in task_ids if tid in task_index), 2
        )
        force_accept_all = iteration >= MAX_ITERATIONS
        lower_relaxed = iteration >= RELAX_LOWER_AFTER

        # Quota check takes priority — a plan without mandatory quota tasks is always invalid
        missing_quota = quota_task_ids - set(task_ids) if has_quota_tasks else set()
        if missing_quota and not force_accept_all:
            feedback = (
                f"Квотные задачи обязательны, но отсутствуют в плане: "
                f"{', '.join(sorted(missing_quota))}. Добавь их."
            )
            validated = False
            verdict = f"ОТКЛОНЕНО: отсутствуют квотные задачи {', '.join(sorted(missing_quota))}"
        elif lower <= total_sp <= upper:
            feedback, validated = "", True
            verdict = f"ПРИНЯТО (идеально): SP={total_sp} в [{lower}, {upper}]"
        elif total_sp > upper and force_accept_all:
            feedback = f"Принято принудительно после {MAX_ITERATIONS} итераций. SP={total_sp}"
            validated = True
            verdict = f"ПРИНЯТО ПРИНУДИТЕЛЬНО: SP={total_sp}"
        elif total_sp > upper:
            feedback = (
                f"Суммарный SP={total_sp} превышает жёсткий лимит {upper} SP "
                f"(110% от ёмкости {capacity}). Необходимо убрать задачи."
            )
            validated = False
            verdict = f"ОТКЛОНЕНО: SP={total_sp} > {upper}"
        elif total_sp < lower and lower_relaxed:
            feedback, validated = "", True
            verdict = (
                f"ПРИНЯТО (недобор после {iteration} итераций): "
                f"SP={total_sp} < {lower} (90%), но итерации исчерпаны"
            )
        else:
            # total_sp < lower, ещё есть итерации
            feedback = (
                f"Суммарный SP={total_sp} ниже целевого {lower} SP "
                f"(90% от ёмкости {capacity}). Добавь задачи из кандидатов."
            )
            validated = False
            verdict = f"ОТКЛОНЕНО: SP={total_sp} < {lower}"

        logger.info(
            "[validator] Iteration %d: %d tasks, SP=%.2f → %s",
            iteration, len(task_ids), total_sp, verdict,
        )
        if feedback and not force_accept_all:
            logger.info("[validator] Feedback for agents: %s", feedback)

        save_critic_iteration_to_redis(
            redis_client=redis_client,
            session_id=session_id,
            team_name=team_name,
            iteration=iteration,
            plan=plan,
            feedback=feedback,
            validated=validated,
            total_sp=total_sp,
        )

        return {**state, "validated": validated, "feedback": feedback}

    async def consultation_node(state: CriticState) -> dict:
        plan = state.get("plan", "")
        iteration = state["iteration"]

        task_ids_in_plan = set(
            state.get("selected_task_ids") or list(dict.fromkeys(_parse_task_ids(plan)))
        )
        total_sp = round(
            sum(task_index[tid].sp for tid in task_ids_in_plan if tid in task_index), 2
        )

        if total_sp > upper:
            sp_status = "over"
            sp_gap = round(total_sp - upper, 2)
            logger.info(
                "[consultation] SP=%.2f, превышение лимита +%.2f SP → запрашиваем что убрать",
                total_sp, sp_gap,
            )
        else:
            sp_status = "under"
            sp_gap = round(lower - total_sp, 2)
            logger.info(
                "[consultation] SP=%.2f, ниже цели на %.2f SP → запрашиваем что добавить",
                total_sp, sp_gap,
            )

        prev_remove = set(state.get("prev_remove_recs") or [])
        prev_add = set(state.get("prev_add_recs") or [])

        specialists = [
            ("inc_agent",     "incident"),
            ("task_agent",    "task"),
            ("project_agent", "project"),
        ]

        # Build filtered task lists and log before firing requests in parallel
        spec_calls: list[tuple[str, Any]] = []
        for specialist_name, category in specialists:
            tasks_selected, tasks_available = _split_tasks(task_ids_in_plan, category)
            if sp_status == "over":
                tasks_selected = [
                    t for t in tasks_selected
                    if not (t.quota and t.quota > 0) and t.task_id not in prev_remove
                ]
                tasks_available = [t for t in tasks_available if t.task_id not in prev_remove]
            else:
                tasks_available = [t for t in tasks_available if t.task_id not in prev_add]

            logger.info(
                "[consultation] → %s (%s): selected=%d, available=%d (skipped prev: %d)",
                specialist_name, sp_status,
                len(tasks_selected), len(tasks_available),
                len(prev_remove if sp_status == "over" else prev_add),
            )
            spec_calls.append((
                specialist_name,
                _consult_specialist(
                    llm=consult_llm,
                    specialist_name=specialist_name,
                    tasks_selected=tasks_selected,
                    tasks_available=tasks_available,
                    sp_status=sp_status,
                    sp_gap=sp_gap,
                ),
            ))

        # Fire all three specialist LLM calls concurrently; failures are logged, not raised
        raw_results = await asyncio.gather(
            *[coro for _, coro in spec_calls], return_exceptions=True
        )

        sections: list[str] = []
        new_remove_recs: list[str] = []
        new_add_recs: list[str] = []

        for (specialist_name, _), advice in zip(spec_calls, raw_results):
            if isinstance(advice, BaseException):
                logger.warning("[%s] consultation failed: %s", specialist_name, advice)
                continue
            if advice:
                logger.info("[%s] recommends:\n%s", specialist_name, advice)
                sections.append(f"=== {specialist_name} ===\n{advice}")
                rec_ids = _extract_known_ids(advice)
                if sp_status == "over":
                    new_remove_recs.extend(rec_ids)
                else:
                    new_add_recs.extend(rec_ids)
            else:
                logger.info("[%s] no recommendations", specialist_name)

        consultation = "\n\n".join(sections) if sections else "Нет рекомендаций от агентов."

        save_critic_consultation_to_redis(
            redis_client=redis_client,
            session_id=session_id,
            team_name=team_name,
            iteration=iteration,
            consultation=consultation,
        )

        return {
            **state,
            "consultation": consultation,
            "prev_remove_recs": list(prev_remove | set(new_remove_recs)),
            "prev_add_recs": list(prev_add | set(new_add_recs)),
        }

    # ── routing ────────────────────────────────────────────────────────────

    def should_continue(state: CriticState) -> str:
        if state.get("validated", False):
            return END
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return END
        return "consultation_node"

    # ── graph ──────────────────────────────────────────────────────────────

    graph = StateGraph(CriticState)
    graph.add_node("critic_node", critic_node)
    graph.add_node("validator_node", validator_node)
    graph.add_node("consultation_node", consultation_node)
    graph.set_entry_point("critic_node")
    graph.add_edge("critic_node", "validator_node")
    graph.add_conditional_edges(
        "validator_node",
        should_continue,
        {"consultation_node": "consultation_node", END: END},
    )
    graph.add_edge("consultation_node", "critic_node")
    return graph.compile()
