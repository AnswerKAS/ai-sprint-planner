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



def make_get_inc_tasks_tool(sprint_task_list: list[SprintTask]):
    @tool
    def get_inc_tasks_temas(team_name: str, runtime: ToolRuntime) -> str:
        """Возвращает инцидентные задачи для команды team_name."""
        store = runtime.store
        current_session_id = runtime.context.session_id

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


def make_get_task_tasks_tool(sprint_task_list: list[SprintTask]):
    @tool
    def get_task_tasks(team_name: str, runtime: ToolRuntime) -> str:
        """Возвращает внутренние задачи для команды team_name."""
        store = runtime.store
        current_session_id = runtime.context.session_id

        task_tasks = [
            task for task in sprint_task_list
            if task.category == "task" and task.team == team_name
        ]

        if store is not None:
            key = f"{current_session_id}:task:{team_name}:last_run"
            value = json.dumps({
                "status": "ok",
                "tasks_count": len(task_tasks),
                "tasks": str([task.model_dump() for task in task_tasks]),
            }, ensure_ascii=False)

            store.mset([(key, value)])

        if not task_tasks:
            return f"Нет внутренних задач для команды {team_name}."

        return f"Внутренние задачи для команды {team_name}:\n" + "\n".join(
            f"- Номер задач: {task.task_id}\n  Описание: {task.title}\n  SP: {task.sp}"
            for task in task_tasks
        )

    return get_task_tasks

@tool
def sum_sp_from_agent_response(agent_response_json: str) -> str:
    """
    Суммирует SP из ответа агента перед финальным ответом для проверки того что выбранные задачи не превышают емкость команды.

    Ожидает JSON-строку в формате:
    {
      "task_list": [
        {
          "task_id": "SA-203",
          "text": "Доработать валидации заявок",
          "sp": 5.0,
          "reasoning": "..."
        }
      ],
      "summary": "..."
    }

    Возвращает JSON-строку:
    {
      "total_sp": 5.0,
      "task_count": 1,
      "task_ids": ["SA-203"]
    }
    """
    try:
        data = json.loads(agent_response_json)
    except Exception as e:
        return json.dumps({
            "error": f"invalid JSON: {str(e)}"
        }, ensure_ascii=False)

    task_list = data.get("task_list")

    if not isinstance(task_list, list):
        return json.dumps({
            "error": "field 'task_list' must be a list"
        }, ensure_ascii=False)

    total_sp = 0.0
    task_ids = []
    errors = []

    for idx, task in enumerate(task_list):
        if not isinstance(task, dict):
            errors.append(f"item at index {idx} is not an object")
            continue

        task_id = task.get("task_id")
        sp = task.get("sp", 0)

        if task_id is not None:
            task_ids.append(task_id)

        try:
            total_sp += float(sp)
        except (TypeError, ValueError):
            errors.append(f"invalid sp for task_id={task_id}")

    result = {
        "total_sp": round(total_sp, 2),
        "task_count": len(task_list),
        "task_ids": task_ids,
    }

    if errors:
        result["errors"] = errors

    return json.dumps(result, ensure_ascii=False)