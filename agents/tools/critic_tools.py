import json

import redis
from langchain.tools import tool

from agents.tools.common_tools import TEAM_CAPACITY
from tasks.model import SprintTask


_AGENT_NAMES = ["inc_agent", "task_agent", "project_agent", "quota_agent"]


def fetch_candidates_context(redis_client: redis.Redis, session_id: str) -> str:
    """Read all specialist agent results from Redis and return as a single context string.
    Called once before the critic loop — result is injected into every prompt directly.
    """
    sections = []
    for agent_name in _AGENT_NAMES:
        key = f"{agent_name}:result:{session_id}"
        raw = redis_client.get(key)
        if raw:
            try:
                data = json.loads(raw)
                content = data.get("result", raw)
            except (json.JSONDecodeError, AttributeError):
                content = str(raw)
        else:
            content = "Нет данных"
        sections.append(f"=== {agent_name} ===\n{content}")
    return "\n\n".join(sections)


def make_critic_sp_tool(
    sprint_task_list: list[SprintTask],
    team_name: str,
    lower_pct: float = 0.9,
    upper_pct: float = 1.1,
):
    task_index = {task.task_id: task for task in sprint_task_list}

    @tool
    def check_sprint_sp(task_ids: list[str]) -> str:
        """Проверяет суммарный SP выбранных задач.
        Идеальный диапазон: 90–110% ёмкости команды.
        Вызывай перед финальным ответом для подтверждения плана.

        Args:
            task_ids: список task_id выбранных задач
        """
        capacity = TEAM_CAPACITY.get(team_name, 0.0)
        lower = round(capacity * lower_pct, 2)
        upper = round(capacity * upper_pct, 2)

        total_sp = 0.0
        breakdown = []
        not_found = []

        for task_id in task_ids:
            task = task_index.get(task_id)
            if task is None:
                not_found.append(task_id)
            else:
                total_sp += task.sp
                breakdown.append({"task_id": task_id, "sp": task.sp})

        total_sp = round(total_sp, 2)
        utilization = round(total_sp / capacity * 100, 1) if capacity > 0 else 0.0

        if total_sp > upper:
            verdict = (
                f"ПРЕВЫШЕНИЕ: SP={total_sp} > лимита {upper} ({upper_pct*100:.0f}% "
                f"от ёмкости {capacity}). Убери задачи."
            )
        elif total_sp >= lower:
            verdict = (
                f"ИДЕАЛЬНО: SP={total_sp} в диапазоне [{lower}, {upper}] "
                f"({lower_pct*100:.0f}–{upper_pct*100:.0f}% от ёмкости {capacity}). "
                f"Загрузка: {utilization}%."
            )
        else:
            verdict = (
                f"НИЖЕ ЦЕЛИ: SP={total_sp} < целевого {lower} ({lower_pct*100:.0f}% "
                f"от ёмкости {capacity}). Загрузка: {utilization}%. Можно добавить задачи."
            )

        result: dict = {
            "total_sp": total_sp,
            "capacity": capacity,
            "lower_target": lower,
            "upper_limit": upper,
            "utilization_pct": utilization,
            "verdict": verdict,
            "breakdown": breakdown,
        }
        if not_found:
            result["not_found"] = not_found

        return json.dumps(result, ensure_ascii=False)

    return check_sprint_sp
