import json
import logging
import os
from typing import Any

import redis

logger = logging.getLogger(__name__)

REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))



def extract_final_response(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response"
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None:
        return str(last)
    return content if isinstance(content, str) else str(content)


def save_agent_result_to_redis(
    redis_client: redis.Redis,
    session_id: str,
    team_name: str,
    result: dict[str, Any],
    agent_name: str,
) -> str:
    payload = {
        "session_id": session_id,
        "team_name": team_name,
        "result": extract_final_response(result),
    }
    redis_key = f"{agent_name}:result:{session_id}"
    redis_client.set(
        redis_key,
        json.dumps(payload, ensure_ascii=False, indent=2),
        ex=REDIS_TTL,
    )
    logger.debug("Saved agent result to Redis: %s", redis_key)
    return redis_key


def create_redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True,
    )


redis_client = create_redis_client()
