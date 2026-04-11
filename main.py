from typing import Any
import json
import uuid

import yaml
import redis
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from agents.inc_agent import init_inc_agent
from agents.model import ResponseAgent
from tasks.loader.excel_loader import load_tasks_from_excel

# Ваш store для create_agent можно оставить, если он нужен агенту
from langchain_community.storage import RedisStore
from store.redis import save_agent_result_to_redis, to_jsonable


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

    # 1) Store, который Вы передаёте в агент
    store = RedisStore(redis_url="redis://:mystrongpassword@localhost:6379/0")

    # 2) Отдельный обычный Redis-клиент для явной записи результата
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        password="mystrongpassword",
        decode_responses=True,
    )

    ai_agents_configs = read_config_ai_agents()

    task_list = load_tasks_from_excel(
        r"C:\Users\anton\Desktop\git\langchain\srpint_ai_agent\sprint_tasks_template_short.xlsx"
    )
    inc_agent = init_inc_agent(ai_agents_configs, task_list, store)

    team_name = "SA"

    result = inc_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Привет, inc_agent! Какие инцидентные задачи ты бы выбрал для работы?\n"
                        f"team_name = \"{team_name}\""
                    ),
                },
            ]
        },
        context={
            "session_id": session_id,
        },
    )

    redis_key = save_agent_result_to_redis(
        redis_client=redis_client,
        session_id=session_id,
        team_name=team_name,
        result=result,
    )

    print("=== RESULT ===")
    print(result)

    print("\n=== SAVED TO REDIS ===")
    print(redis_key)

    print("\n=== STRUCTURED RESPONSE ===")
    print(to_jsonable(result.get("structured_response")))