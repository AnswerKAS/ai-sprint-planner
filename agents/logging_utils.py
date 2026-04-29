import logging
from typing import Any

logger = logging.getLogger(__name__)

_CONTENT_MAX = 600
_ARGS_MAX = 200


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars]"


def log_agent_messages(messages: list[Any], agent_name: str) -> None:
    """Log the full tool-call / response chain of a LangChain agent invocation."""
    logger.info("┌── %s | decisions ─────────────────────────────────", agent_name)
    for msg in messages:
        msg_type = type(msg).__name__
        content = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", []) or []
        name = getattr(msg, "name", None)

        if msg_type == "HumanMessage":
            if isinstance(content, str) and content.strip():
                logger.info("│ [USER] %s", _truncate(content, _CONTENT_MAX))

        elif msg_type == "AIMessage":
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc.get("name", "?")
                    tc_args = str(tc.get("args", {}))
                    logger.info(
                        "│ [TOOL CALL] %s(%s)", tc_name, _truncate(tc_args, _ARGS_MAX)
                    )
            elif isinstance(content, str) and content.strip():
                formatted = content.replace("\n", "\n│   ")
                logger.info("│ [RESPONSE]\n│   %s", _truncate(formatted, _CONTENT_MAX))

        elif msg_type == "ToolMessage":
            label = name or "tool"
            if isinstance(content, str):
                logger.info(
                    "│ [TOOL RESULT: %s] %s", label, _truncate(content, _CONTENT_MAX)
                )

    logger.info("└───────────────────────────────────────────────────────")
