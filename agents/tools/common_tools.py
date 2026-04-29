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
            f"\n  SP: {task.sp} | Приоритет: {task.priority} | RICE: {task.rice}"
            f" | Эскалации: {task.escalation_count} | Бизнес-ценность: {task.business_value}"
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