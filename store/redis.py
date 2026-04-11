import json
from typing import Any

import redis


def _last_message_content(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response"
    last = messages[-1]
    return getattr(last, "content", str(last))


def to_jsonable(value: Any) -> Any:
    """Преобразует результат агента в JSON-совместимый вид."""
    if value is None:
        return None

    if hasattr(value, "model_dump"):  # Pydantic v2
        return value.model_dump()

    if hasattr(value, "dict"):  # Pydantic v1 / прочие модели
        return value.dict()

    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    return value



def save_agent_result_to_redis(
    redis_client: redis.Redis,
    session_id: str,
    team_name: str,
    result: dict[str, Any],
) -> str:
    """
    Сохраняет итог выполнения агента в Redis.
    Возвращает ключ, по которому запись сохранена.
    """
    structured_response = result.get("structured_response")
    last_message = _last_message_content(result)

    payload = {
        "session_id": session_id,
        "team_name": team_name,
        "structured_response": to_jsonable(structured_response),
        "last_message": last_message,
        "raw_result": to_jsonable(result),
    }

    redis_key = f"inc_agent:result:{session_id}"
    redis_client.set(redis_key, json.dumps(payload, ensure_ascii=False, indent=2), ex=300)
    return redis_key


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    password="mystrongpassword",
    decode_responses=True,
)
