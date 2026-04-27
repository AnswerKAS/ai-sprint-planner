import json
import uuid
from typing import Any, Callable

import redis as redis_lib
from langchain.tools import tool
from langchain_community.storage import RedisStore

from agents.model import OrbitSprintPlan, OrbitTask
from agents.tools.common_tools import TEAM_CAPACITY
from tasks.model import SprintTask

MAX_ITERATIONS = 5


def _validate_plan(team_name: str, task_ids: list[str], task_index: dict) -> dict:
    capacity = TEAM_CAPACITY.get(team_name, 0.0)
    max_allowed = round(capacity * 1.05, 2)
    min_required = round(capacity * 0.85, 2)

    total_sp = 0.0
    breakdown = []
    not_found = []

    for tid in task_ids:
        task = task_index.get(tid)
        if task is None:
            not_found.append(tid)
            continue
        total_sp += task.sp
        breakdown.append({"task_id": tid, "sp": task.sp, "title": task.title, "rice": task.rice or 0})

    total_sp = round(total_sp, 2)
    deficit = round(max(0.0, min_required - total_sp), 2)
    fill_pct = round((total_sp / capacity * 100) if capacity > 0 else 0.0, 1)
    is_over = total_sp > max_allowed
    is_under = total_sp < min_required
    # Дефицит < 1 SP несущественен — при отсутствии кандидатов можно принять план
    near_valid = is_under and deficit < 1.0
    is_valid = not is_over and not is_under

    if is_over:
        verdict = (
            f"ПРЕВЫШЕНИЕ: SP={total_sp} > максимум={max_allowed} ({fill_pct}%). "
            f"Убери задачи (только не-quota)."
        )
    elif near_valid:
        verdict = (
            f"ПОЧТИ ВАЛИДЕН: SP={total_sp}, дефицит={deficit} SP (< 1 SP, {fill_pct}% ёмкости). "
            f"Если all_candidates_exhausted=true — переходи к ШАГу 5."
        )
    elif is_under:
        verdict = (
            f"НЕДОБОР: SP={total_sp} < минимум={min_required} ({fill_pct}%), дефицит={deficit} SP. "
            f"Добавь задачи из remaining_candidates или через consult_sub_agent."
        )
    else:
        verdict = f"ПЛАН ВАЛИДЕН: SP={total_sp} в диапазоне [{min_required}–{max_allowed}] ({fill_pct}%)."

    result: dict = {
        "total_sp": total_sp,
        "capacity": capacity,
        "max_allowed": max_allowed,
        "min_required": min_required,
        "deficit": deficit,
        "fill_pct": fill_pct,
        "is_over": is_over,
        "is_under": is_under,
        "near_valid": near_valid,
        "is_valid": is_valid,
        "verdict": verdict,
    }
    if not_found:
        result["not_found"] = not_found
    return result


def make_orbit_plan_tools(
    redis_client: redis_lib.Redis,
    session_id: str,
    task_list: list[SprintTask],
):
    """
    Возвращает два инструмента (build_initial_plan, update_sprint_plan), которые
    разделяют общее изменяемое состояние плана, счётчик итераций и историю изменений.
    """
    task_index = {task.task_id: task for task in task_list}

    plan_state: dict = {
        "task_ids": [],        # текущий список task_id в плане
        "quota_ids": set(),    # защищённые quota task_id (нельзя удалять)
        "all_candidates": [],  # все не-quota кандидаты от агентов (source, rice, sp)
        "task_history": {},    # task_id -> {iteration, source_agent}
        "iteration": 0,
        "team_name": "",       # заполняется при первом вызове build_initial_plan
    }

    @tool
    def build_initial_plan(team_name: str) -> str:
        """
        ПЕРВЫЙ инструмент при формировании плана. Всегда вызывай его первым.

        Читает результаты всех агентов из Redis и детерминированно строит начальный план:
        1. Включает ВСЕ задачи от quota_agent (iteration=0, обязательные).
        2. Жадно добавляет задачи от inc_agent / task_agent / project_agent по убыванию RICE
           (iteration=0), пока суммарный SP не достигнет 105% ёмкости.

        Возвращает JSON:
          planned_task_ids       — список task_id в плане
          quota_task_ids         — список task_id quota задач (защищены от удаления)
          task_history           — для каждого task_id: {iteration, source_agent}
          by_agent               — задачи плана по агентам (sp, rice)
          remaining_candidates   — задачи агентов, НЕ вошедшие в план (можно добавить)
          all_candidates_exhausted — true если remaining_candidates пуст
          validation             — total_sp, fill_pct, is_valid, is_over, is_under,
                                   near_valid, deficit, verdict
          quota_overflow         — true если quota SP сам по себе > 105% ёмкости
          remaining_iterations   — сколько вызовов update_sprint_plan осталось (макс. 5)
        """
        plan_state["team_name"] = team_name
        capacity = TEAM_CAPACITY.get(team_name, 0.0)
        max_allowed = round(capacity * 1.05, 2)

        raw_by_agent: dict[str, list[dict]] = {}
        for agent_name in ["quota_agent", "inc_agent", "task_agent", "project_agent"]:
            key = f"{agent_name}:result:{session_id}"
            raw = redis_client.get(key)
            if not raw:
                raw_by_agent[agent_name] = []
                continue
            try:
                data = json.loads(raw)
                resp = data.get("structured_response") or data.get("final_response")
                raw_by_agent[agent_name] = resp.get("task_list", []) if isinstance(resp, dict) else []
            except Exception:
                raw_by_agent[agent_name] = []

        planned_ids: list[str] = []
        quota_ids: set[str] = set()
        task_history: dict[str, dict] = {}
        total_sp = 0.0
        seen_ids: set[str] = set()
        by_agent: dict[str, list[dict]] = {
            "quota_agent": [], "inc_agent": [], "task_agent": [], "project_agent": []
        }

        # Шаг 1: все quota задачи — обязательны (iteration=0)
        for td in raw_by_agent.get("quota_agent", []):
            tid = td.get("task_id", "")
            if not tid or tid in seen_ids:
                continue
            task = task_index.get(tid)
            if task:
                planned_ids.append(tid)
                quota_ids.add(tid)
                total_sp += task.sp
                seen_ids.add(tid)
                task_history[tid] = {"iteration": 0, "source_agent": "quota_agent"}
                by_agent["quota_agent"].append(
                    {"task_id": tid, "sp": task.sp, "rice": task.rice or 0}
                )

        # Шаг 2: кандидаты от других агентов — сортируем по RICE убывающе
        candidates: list[dict] = []
        candidate_ids: set[str] = set()
        for agent_name in ["inc_agent", "task_agent", "project_agent"]:
            for td in raw_by_agent.get(agent_name, []):
                tid = td.get("task_id", "")
                if not tid or tid in seen_ids or tid in candidate_ids:
                    continue
                task = task_index.get(tid)
                if task:
                    candidates.append({
                        "task_id": tid,
                        "sp": task.sp,
                        "text": task.title,
                        "rice": task.rice or 0.0,
                        "source": agent_name,
                    })
                    candidate_ids.add(tid)

        candidates.sort(key=lambda c: c["rice"], reverse=True)

        # Жадное заполнение до max_allowed (iteration=0)
        for cand in candidates:
            if total_sp + cand["sp"] <= max_allowed:
                tid = cand["task_id"]
                planned_ids.append(tid)
                total_sp += cand["sp"]
                seen_ids.add(tid)
                task_history[tid] = {"iteration": 0, "source_agent": cand["source"]}
                by_agent[cand["source"]].append(
                    {"task_id": tid, "sp": cand["sp"], "rice": cand["rice"]}
                )

        # Кандидаты, не вошедшие в план — для добора
        remaining_candidates: list[dict] = [
            {"task_id": c["task_id"], "sp": c["sp"], "rice": c["rice"], "source": c["source"]}
            for c in candidates if c["task_id"] not in seen_ids
        ]

        # Инициализируем состояние
        plan_state["task_ids"] = planned_ids[:]
        plan_state["quota_ids"] = quota_ids
        plan_state["all_candidates"] = candidates[:]
        plan_state["task_history"] = task_history
        plan_state["iteration"] = 0

        validation = _validate_plan(team_name, planned_ids, task_index)

        quota_sp = round(sum(task_index[tid].sp for tid in quota_ids if tid in task_index), 2)
        quota_overflow = quota_sp > max_allowed

        if quota_overflow:
            validation["verdict"] = (
                f"QUOTA_OVERFLOW: quota задачи (SP={quota_sp}) превышают 105% ёмкости "
                f"(max={max_allowed}). Неустранимо. Переходи сразу к ШАГу 5."
            )
            validation["is_over"] = True
            validation["is_valid"] = False

        return json.dumps({
            "planned_task_ids": planned_ids,
            "quota_task_ids": list(quota_ids),
            "task_history": task_history,
            "by_agent": by_agent,
            "remaining_candidates": remaining_candidates,
            "all_candidates_exhausted": len(remaining_candidates) == 0,
            "validation": validation,
            "quota_overflow": quota_overflow,
            "remaining_iterations": MAX_ITERATIONS,
        }, ensure_ascii=False, indent=2)

    @tool
    def update_sprint_plan(
        team_name: str,
        add_task_ids: list[str],
        remove_task_ids: list[str],
    ) -> str:
        """
        Обновляет текущий план: добавляет и/или удаляет задачи. Расходует одну итерацию.
        Quota задачи защищены — удалить их невозможно.

        Args:
            team_name: название команды
            add_task_ids: task_id для добавления (пустой список — пропустить)
            remove_task_ids: task_id для удаления (quota игнорируются)

        Возвращает JSON:
          current_plan_task_ids  — текущий список task_id после изменений
          quota_task_ids         — защищённые task_id
          task_history           — для каждого task_id в плане: {iteration, source_agent}
          remaining_candidates   — задачи ещё не в плане (можно добавить)
          all_candidates_exhausted — true если remaining_candidates пуст
          iteration              — номер текущей итерации
          remaining_iterations   — сколько итераций осталось
          changes                — добавлено / удалено / отклонено
          validation             — новая валидация (total_sp, fill_pct, is_valid, near_valid, ...)
          warning                — если итерации исчерпаны
        """
        plan_state["iteration"] += 1
        iteration = plan_state["iteration"]
        remaining = MAX_ITERATIONS - iteration

        current_set = set(plan_state["task_ids"])
        protected = plan_state["quota_ids"]
        candidate_source_map = {c["task_id"]: c["source"] for c in plan_state["all_candidates"]}

        added: list[str] = []
        rejected_add: list[dict] = []
        removed: list[str] = []
        protected_from_removal: list[str] = []

        for tid in add_task_ids:
            if tid in current_set:
                rejected_add.append({"task_id": tid, "reason": "уже в плане"})
                continue
            task = task_index.get(tid)
            if task is None:
                rejected_add.append({"task_id": tid, "reason": "задача не найдена"})
                continue
            plan_state["task_ids"].append(tid)
            current_set.add(tid)
            added.append(tid)
            plan_state["task_history"][tid] = {
                "iteration": iteration,
                "source_agent": candidate_source_map.get(tid, "consult_sub_agent"),
            }

        for tid in remove_task_ids:
            if tid in protected:
                protected_from_removal.append(tid)
                continue
            if tid in current_set:
                plan_state["task_ids"].remove(tid)
                current_set.discard(tid)
                removed.append(tid)
                plan_state["task_history"].pop(tid, None)

        remaining_candidates = [
            {"task_id": c["task_id"], "sp": c["sp"], "rice": c["rice"], "source": c["source"]}
            for c in plan_state["all_candidates"]
            if c["task_id"] not in current_set
        ]

        validation = _validate_plan(team_name, plan_state["task_ids"], task_index)

        changes: dict = {"added": added, "removed": removed}
        if rejected_add:
            changes["rejected_add"] = rejected_add
        if protected_from_removal:
            changes["protected_from_removal"] = protected_from_removal
            changes["protection_note"] = (
                f"Задачи {protected_from_removal} — quota-задачи, удалить нельзя. "
                f"Удаляй только задачи из by_agent.inc_agent / task_agent / project_agent."
            )

        result: dict = {
            "current_plan_task_ids": plan_state["task_ids"][:],
            "quota_task_ids": list(plan_state["quota_ids"]),
            "task_history": plan_state["task_history"],
            "remaining_candidates": remaining_candidates,
            "all_candidates_exhausted": len(remaining_candidates) == 0,
            "iteration": iteration,
            "remaining_iterations": remaining,
            "changes": changes,
            "validation": validation,
        }

        if remaining <= 0 and not validation["is_valid"]:
            result["warning"] = (
                f"Лимит итераций ({MAX_ITERATIONS}) исчерпан. "
                f"Верни план как есть: SP={validation['total_sp']}, {validation['fill_pct']}% ёмкости."
            )

        return json.dumps(result, ensure_ascii=False, indent=2)

    return build_initial_plan, update_sprint_plan, plan_state


def build_orbit_sprint_plan(plan_state: dict, task_index: dict) -> OrbitSprintPlan:
    """Строит OrbitSprintPlan из plan_state после завершения агента."""
    team_name = plan_state.get("team_name", "")
    task_ids = plan_state.get("task_ids", [])
    task_history = plan_state.get("task_history", {})

    validation = _validate_plan(team_name, task_ids, task_index)

    tasks: list[OrbitTask] = []
    for tid in task_ids:
        task = task_index.get(tid)
        if not task:
            continue
        history = task_history.get(tid, {})
        iteration = history.get("iteration", 0)
        source = history.get("source_agent", "unknown")

        if source == "quota_agent":
            reasoning = "Обязательная quota-задача (quota > 0), включена безусловно"
        elif iteration == 0:
            reasoning = f"Включена в начальный план агентом {source} по наибольшему RICE"
        else:
            reasoning = f"Добавлена на итерации {iteration} агентом {source} для доведения SP до целевого диапазона"

        tasks.append(OrbitTask(
            task_id=tid,
            text=task.title,
            sp=task.sp,
            source_agent=source,
            iteration=iteration,
            reasoning=reasoning,
        ))

    by_type: dict[str, int] = {}
    for h in task_history.values():
        src = h.get("source_agent", "unknown")
        by_type[src] = by_type.get(src, 0) + 1

    type_parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
    is_valid = validation["is_valid"] or validation.get("near_valid", False)
    summary = (
        f"Итоговый план: SP={validation['total_sp']}, {validation['fill_pct']}% ёмкости "
        f"({validation['total_sp']}/{validation['capacity']} SP), "
        f"{len(tasks)} задач ({type_parts})"
    )

    return OrbitSprintPlan(
        team_name=team_name,
        task_list=tasks,
        total_sp=validation["total_sp"],
        fill_pct=validation["fill_pct"],
        is_valid=is_valid,
        summary=summary,
    )


def make_consult_sub_agent_tool(
    agent_factories: dict[str, Callable],
    all_configs: dict[str, Any],
    task_list: list[SprintTask],
    store: RedisStore,
    session_id: str | None,
):
    @tool
    def consult_sub_agent(
        agent_name: str,
        team_name: str,
        request: str,
        excluded_task_ids: list[str],
    ) -> str:
        """
        Обращается к дочернему агенту (task_agent / inc_agent / project_agent).
        НЕ расходует итерацию — используй до вызова update_sprint_plan.

        Для is_under (добор):
          excluded_task_ids = все task_id УЖЕ в плане → агент предложит задачи из оставшихся.

        Для is_over (удаление):
          excluded_task_ids = task_id задач этого агента НЕ в плане →
          агент видит только свои задачи из плана и выбирает что ОСТАВИТЬ.
          Задачи, которые агент не вернул → кандидаты на удаление через update_sprint_plan.

        Args:
            agent_name: "task_agent", "inc_agent" или "project_agent"
            team_name: название команды
            request: запрос агенту
            excluded_task_ids: task_id задач, скрытых от агента

        Returns:
            JSON с полями team_name, task_list (предложения агента), summary
        """
        factory = agent_factories.get(agent_name)
        if factory is None:
            return json.dumps({
                "error": f"Неизвестный агент: {agent_name}. Допустимые: task_agent, inc_agent, project_agent."
            })

        config = all_configs.get(agent_name, {})
        excluded_set = set(excluded_task_ids)
        filtered_tasks = [t for t in task_list if t.task_id not in excluded_set]

        sub_session_id = f"{session_id}:sub:{agent_name}:{uuid.uuid4().hex[:6]}"
        sub_agent = factory(config, filtered_tasks, store, session_id=sub_session_id)

        prompt = f'team_name = "{team_name}"\n{request}'
        result = sub_agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        structured = result.get("structured_response")
        if structured is not None:
            if hasattr(structured, "model_dump"):
                return json.dumps(structured.model_dump(), ensure_ascii=False, indent=2)
            if isinstance(structured, dict):
                return json.dumps(structured, ensure_ascii=False, indent=2)

        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", str(last))
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

        return json.dumps({"error": "Нет ответа от агента"})

    return consult_sub_agent
