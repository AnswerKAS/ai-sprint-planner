import json
from typing import Any

import redis


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


def extract_final_response(result: dict[str, Any]) -> Any:
    """
    Возвращает нормализованный итог ответа агента.
    Приоритет:
    1. structured_response
    2. последнее сообщение messages[-1]
    """
    structured = result.get("structured_response")
    if structured is not None:
        return to_jsonable(structured)

    messages = result.get("messages", [])
    if not messages:
        return "No response"

    last = messages[-1]
    content = getattr(last, "content", None)

    if content is None:
        return str(last)

    if isinstance(content, (dict, list)):
        if isinstance(content, dict) and "name" in content and "arguments" in content:
            return content["arguments"]
        return content

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                return parsed["arguments"]
            return parsed
        except Exception:
            return content

    return str(content)


def save_agent_result_to_redis(
    redis_client: redis.Redis,
    session_id: str,
    team_name: str,
    result: dict[str, Any],
    agent_name: str
) -> str:
    """
    Сохраняет итог выполнения агента в Redis.
    Возвращает ключ, по которому запись сохранена.
    """
    payload = {
        "session_id": session_id,
        "team_name": team_name,
        "final_response": extract_final_response(result),
        "structured_response": to_jsonable(result.get("structured_response")),
        "raw_result": to_jsonable(result),
    }

    redis_key = f"{agent_name}:result:{session_id}"
    redis_client.set(
        redis_key,
        json.dumps(payload, ensure_ascii=False, indent=2),
        ex=300
    )
    return redis_key


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    password="mystrongpassword",
    decode_responses=True,
)