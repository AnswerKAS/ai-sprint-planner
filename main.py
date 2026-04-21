from typing import Any
import json
import uuid

import yaml
import redis
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from agents.inc_agent import init_inc_agent
from agents.model import ResponseAgent
from agents.task_agent import init_task_agent
from tasks.loader.excel_loader import load_tasks_from_excel

# Ваш store для create_agent можно оставить, если он нужен агенту
from langchain_community.storage import RedisStore
from store.redis import save_agent_result_to_redis, to_jsonable, redis_client

from agents.agent_builder import agent_builder

session_id = None


def init_session():
    global session_id
    session_id = str(uuid.uuid4())


def _last_message_content(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response"
    last = messages[-1]
    return getattr(last, "content", str(last))


### Чтение конфигурации AI агентов из YAML файла
def read_config_ai_agents() -> dict:
    with open("./agents/promts_agent.yaml", "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


if __name__ == "__main__":
    init_session()
    ai_agents_configs = read_config_ai_agents()

    store = RedisStore(redis_url="redis://:mystrongpassword@localhost:6379/0")
    task_list = load_tasks_from_excel("sprint_tasks_template_short.xlsx")

    team_name = "SA"

    builder_inc = agent_builder(
        agent=init_inc_agent,
        prompt=(
            "Привет, inc_agent! Какие инцидентные задачи ты бы выбрал для работы?\n"
            f'team_name = "{team_name}"'
        ),
        task_list=task_list,
        config=ai_agents_configs.get("inc_agent", {}),
        store=store,
        session_id=session_id,
        redis_client=redis_client,
        team_name=team_name,
        save_to_redis=True,
    )

    builder_task = agent_builder(
        agent=init_task_agent,
        prompt=(
            "Привет, task_agent! Какие внутренние задачи ты бы выбрал для работы?\n"
            f'team_name = "{team_name}"'
        ),
        task_list=task_list,
        config=ai_agents_configs.get("task_agent", {}),
        store=store,
        session_id=session_id,
        redis_client=redis_client,
        team_name=team_name,
        save_to_redis=True,
    )


    inc_agent = builder_inc["agent_instance"]
    result_inc = builder_inc["result"]
    redis_key = builder_inc["redis_key"]


    task_agent = builder_task["agent_instance"]
    result_task = builder_task["result"]
    redis_key_task = builder_task["redis_key"]



    print("=== inc_agent ===")
    print(inc_agent)


    print("\n=== result ===")
    print(result_inc)

    print("\n=== redis_key ===")
    print(redis_key)

    print("=== task_agent ===")
    print(task_agent)

    print("\n=== result ===")
    print(result_task)

    print("\n=== redis_key ===")
    print(redis_key_task)