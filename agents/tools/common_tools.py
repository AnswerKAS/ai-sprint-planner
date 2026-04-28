import json

from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime

from tasks.model import SprintTask



TEAM_CAPACITY: dict[str, float] = {
    "Python": 45.0,
    "SA": 45.0,
    "Meth": 45.0,
}

TEAM_MEMBERS: dict[str, int] = {
    "Python": 3,
    "SA": 3,
    "Meth": 3,
}




@tool
def get_team_capacity(team_name: str) -> str:
    """Возвращает емкость команды в SP."""
    capacity = TEAM_CAPACITY.get(team_name, 0)
    return f"{team_name}: {capacity} SP"


@tool
def get_team_members(team_name: str) -> str:
    """Возвращает количество сотрудников в команде."""
    members = TEAM_MEMBERS.get(team_name, 0)
    return f"{team_name}: {members} сотрудника(ов)"



def make_get_inc_tasks_tool(sprint_task_list: list[SprintTask], session_id: str | None = None):
    @tool
    def get_inc_tasks_teams(team_name: str, runtime: ToolRuntime) -> str:
        """Возвращает инцидентные задачи для команды team_name."""
        store = runtime.store

        inc_tasks = [
            task for task in sprint_task_list
            if task.category == "incident" and task.team == team_name
        ]

        if store is not None and session_id is not None:
            key = f"{session_id}:inc:{team_name}:last_run"
            value = json.dumps({
                "status": "ok",
                "tasks_count": len(inc_tasks),
                "tasks": [task.model_dump() for task in inc_tasks],
            }, ensure_ascii=False)
            store.mset([(key, value)])

        if not inc_tasks:
            return f"Нет инцидентных задач для команды {team_name}."

        return f"Инцидентные задачи для команды {team_name}:\n" + "\n".join(
            f"- {task.task_id}: {task.title}"
            f"\n  SP: {task.sp} | Приоритет: {task.priority} | Эскалации: {task.escalation_count} | Бизнес-ценность: {task.business_value}"
            for task in inc_tasks
        )

    return get_inc_tasks_teams


def make_get_task_tasks_tool(sprint_task_list: list[SprintTask], session_id: str | None = None):
    @tool
    def get_task_tasks(team_name: str, runtime: ToolRuntime) -> str:
        """Возвращает внутренние задачи для команды team_name."""
        store = runtime.store

        task_tasks = [
            task for task in sprint_task_list
            if task.category == "task" and task.team == team_name
        ]

        if store is not None and session_id is not None:
            key = f"{session_id}:task:{team_name}:last_run"
            value = json.dumps({
                "status": "ok",
                "tasks_count": len(task_tasks),
                "tasks": [task.model_dump() for task in task_tasks],
            }, ensure_ascii=False)
            store.mset([(key, value)])

        if not task_tasks:
            return f"Нет внутренних задач для команды {team_name}."

        return f"Внутренние задачи для команды {team_name}:\n" + "\n".join(
            f"- {task.task_id}: {task.title}"
            f"\n  SP: {task.sp} | Приоритет: {task.priority} | RICE: {task.rice} | Бизнес-ценность: {task.business_value}"
            for task in task_tasks
        )

    return get_task_tasks


def make_get_project_tasks_tool(sprint_task_list: list[SprintTask], session_id: str | None = None):
    @tool
    def get_project_tasks(team_name: str, runtime: ToolRuntime) -> str:
        """Возвращает проектные задачи для команды team_name."""
        store = runtime.store

        project_tasks = [
            task for task in sprint_task_list
            if task.category == "project" and task.team == team_name
        ]

        if store is not None and session_id is not None:
            key = f"{session_id}:project:{team_name}:last_run"
            value = json.dumps({
                "status": "ok",
                "tasks_count": len(project_tasks),
                "tasks": [task.model_dump() for task in project_tasks],
            }, ensure_ascii=False)
            store.mset([(key, value)])

        if not project_tasks:
            return f"Нет проектных задач для команды {team_name}."

        return f"Проектные задачи для команды {team_name}:\n" + "\n".join(
            f"- {task.task_id}: {task.title}"
            f"\n  SP: {task.sp} | Приоритет: {task.priority} | RICE: {task.rice} | Бизнес-ценность: {task.business_value} | Стадия: {task.stage} | Эскалации: {task.escalation_count}"
            for task in project_tasks
        )

    return get_project_tasks


def make_get_quota_tasks_tool(sprint_task_list: list[SprintTask], session_id: str | None = None):
    @tool
    def get_quota_tasks(team_name: str, runtime: ToolRuntime) -> str:
        """Возвращает задачи, идущие по квоте (quota > 0), для команды team_name."""
        store = runtime.store

        quota_tasks = [
            task for task in sprint_task_list
            if task.team == team_name and task.quota is not None and task.quota > 0
        ]

        if store is not None and session_id is not None:
            key = f"{session_id}:quota:{team_name}:last_run"
            value = json.dumps({
                "status": "ok",
                "tasks_count": len(quota_tasks),
                "tasks": [task.model_dump() for task in quota_tasks],
            }, ensure_ascii=False)
            store.mset([(key, value)])

        if not quota_tasks:
            return f"Нет квотных задач для команды {team_name}."

        return f"Квотные задачи для команды {team_name}:\n" + "\n".join(
            f"- {task.task_id}: {task.title}"
            f"\n  SP: {task.sp} | Квота: {task.quota} | Приоритет: {task.priority} | Бизнес-ценность: {task.business_value}"
            for task in quota_tasks
        )

    return get_quota_tasks


def make_calculate_sp_tool(sprint_task_list: list[SprintTask]):
    task_index = {task.task_id: task for task in sprint_task_list}

    @tool
    def calculate_selected_sp(team_name: str, task_ids: list[str]) -> str:
        """
        Суммирует SP по выбранным задачам и сравнивает с ёмкостью команды.
        Вызывай этот инструмент ПЕРЕД финальным ответом, чтобы убедиться,
        что суммарный SP не превышает ёмкость команды.

        Args:
            team_name: название команды (например "SA", "Python", "Meth")
            task_ids: список идентификаторов выбранных задач

        Возвращает JSON с полями:
          total_sp       — суммарный SP выбранных задач
          capacity       — ёмкость команды в SP
          remaining_sp   — оставшаяся ёмкость после выбранных задач
          is_over        — true если суммарный SP превышает ёмкость
          verdict        — текстовое заключение
          breakdown      — SP по каждой задаче
          not_found      — task_id, которые не найдены в списке задач
        """
        capacity = TEAM_CAPACITY.get(team_name, 0.0)

        total_sp = 0.0
        breakdown = []
        not_found = []

        for task_id in task_ids:
            task = task_index.get(task_id)
            if task is None:
                not_found.append(task_id)
                continue
            total_sp += task.sp
            breakdown.append({"task_id": task_id, "sp": task.sp, "title": task.title})

        total_sp = round(total_sp, 2)
        remaining_sp = round(capacity - total_sp, 2)
        is_over = total_sp > capacity

        if is_over:
            verdict = (
                f"ПРЕВЫШЕНИЕ ЁМКОСТИ: суммарный SP={total_sp} превышает ёмкость команды {team_name}={capacity} SP "
                f"на {round(total_sp - capacity, 2)} SP. Необходимо убрать часть задач."
            )
        else:
            verdict = (
                f"В РАМКАХ ЁМКОСТИ: суммарный SP={total_sp} не превышает ёмкость команды {team_name}={capacity} SP. "
                f"Остаток: {remaining_sp} SP."
            )

        result: dict = {
            "total_sp": total_sp,
            "capacity": capacity,
            "remaining_sp": remaining_sp,
            "is_over": is_over,
            "verdict": verdict,
            "breakdown": breakdown,
        }
        if not_found:
            result["not_found"] = not_found

        return json.dumps(result, ensure_ascii=False)

    return calculate_selected_sp