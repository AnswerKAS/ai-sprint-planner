import json

from langchain.tools import tool
from langchain_community.storage import RedisStore
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



def make_get_inc_tasks_tool(sprint_task_list: list[SprintTask]):
    @tool
    def get_inc_tasks_temas(team_name: str, runtime: ToolRuntime) -> str:
        """Возвращает инцидентные задачи для команды team_name."""
        store = runtime.store
        current_session_id = runtime.context.get("session_id")

        inc_tasks = [
            task for task in sprint_task_list
            if task.category == "incident" and task.team == team_name
        ]

        if store is not None:
            key = f"{current_session_id}:inc:{team_name}:last_run"
            value = json.dumps({
                "status": "ok",
                "tasks_count": len(inc_tasks),
                "tasks": str([task.model_dump() for task in inc_tasks]),
            }, ensure_ascii=False)

            store.mset([(key, value)])

        if not inc_tasks:
            return f"Нет инцидентных задач для команды {team_name}."

        return f"Инцидентные задачи для команды {team_name}:\n" + "\n".join(
            f"- Номер задач: {task.task_id}\n  Описание: {task.title}\n  SP: {task.sp}"
            for task in inc_tasks
        )

    return get_inc_tasks_temas