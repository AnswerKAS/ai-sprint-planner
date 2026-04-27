from typing import Any, Callable, Optional

from langchain_community.storage import RedisStore

from store.redis import save_agent_result_to_redis
from tasks.model import SprintTask


def agent_builder(
    agent: Callable,
    *,
    prompt: str,
    task_list: list[SprintTask],
    config: dict,
    store: Optional[RedisStore] = None,
    session_id: Optional[str] = None,
    redis_client: Any = None,
    team_name: Optional[str] = None,
    save_to_redis: bool = False,
    **kwargs,
) -> dict[str, Any]:
    agent_name = config.get("name", "agent")
    agent_instance = agent(config, task_list, store, session_id=session_id, **kwargs)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }

    result = agent_instance.invoke(payload)

    redis_key = None
    if save_to_redis:
        if redis_client is None:
            raise ValueError("Для save_to_redis=True необходимо передать redis_client")
        if session_id is None:
            raise ValueError("Для save_to_redis=True необходимо передать session_id")
        if team_name is None:
            raise ValueError("Для save_to_redis=True необходимо передать team_name")

        redis_key = save_agent_result_to_redis(
            redis_client=redis_client,
            session_id=session_id,
            team_name=team_name,
            result=result,
            agent_name=agent_name,
        )

    return {
        "agent_instance": agent_instance,
        "result": result,
        "redis_key": redis_key,
    }
