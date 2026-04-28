import logging
import os
import uuid
from typing import Any

import yaml
from langchain_community.storage import RedisStore

from agents.agent_builder import agent_builder
from agents.agent_factory import init_inc_agent, init_project_agent, init_quota_agent, init_task_agent
from store.redis import REDIS_TTL, redis_client
from tasks.loader.excel_loader import load_tasks_from_excel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _read_agent_configs() -> dict:
    with open("./agents/promts_agent.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _last_message_content(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response"
    last = messages[-1]
    return getattr(last, "content", str(last))


def main() -> None:
    session_id = str(uuid.uuid4())
    logger.info("Session started: %s", session_id)

    configs = _read_agent_configs()
    task_list = load_tasks_from_excel("sprint_tasks_template_short.xlsx")

    redis_url = (
        f"redis://:{os.getenv('REDIS_PASSWORD', '')}@"
        f"{os.getenv('REDIS_HOST', 'localhost')}:"
        f"{os.getenv('REDIS_PORT', '6379')}/0"
    )
    store = RedisStore(redis_url=redis_url, ttl=REDIS_TTL)

    team_name = "SA"

    builder_inc = agent_builder(
        agent=init_inc_agent,
        prompt=f"Какие инцидентные задачи взять в работу?\nteam_name = \"{team_name}\"",
        task_list=task_list,
        config=configs.get("inc_agent", {}),
        store=store,
        session_id=session_id,
        redis_client=redis_client,
        team_name=team_name,
        save_to_redis=True,
    )

    builder_task = agent_builder(
        agent=init_task_agent,
        prompt=f"Какие внутренние задачи взять в работу?\nteam_name = \"{team_name}\"",
        task_list=task_list,
        config=configs.get("task_agent", {}),
        store=store,
        session_id=session_id,
        redis_client=redis_client,
        team_name=team_name,
        save_to_redis=True,
    )

    builder_project = agent_builder(
        agent=init_project_agent,
        prompt=f"Какие проектные задачи взять в работу?\nteam_name = \"{team_name}\"",
        task_list=task_list,
        config=configs.get("project_agent", {}),
        store=store,
        session_id=session_id,
        redis_client=redis_client,
        team_name=team_name,
        save_to_redis=True,
    )

    builder_quota = agent_builder(
        agent=init_quota_agent,
        prompt=f"Какие квотные задачи взять в работу?\nteam_name = \"{team_name}\"",
        task_list=task_list,
        config=configs.get("quota_agent", {}),
        store=store,
        session_id=session_id,
        redis_client=redis_client,
        team_name=team_name,
        save_to_redis=True,
    )

    logger.info("inc_agent redis_key: %s", builder_inc["redis_key"])
    logger.info("inc_agent result: %s", builder_inc["result"])

    logger.info("task_agent redis_key: %s", builder_task["redis_key"])
    logger.info("task_agent result: %s", builder_task["result"])

    logger.info("project_agent redis_key: %s", builder_project["redis_key"])
    logger.info("project_agent result: %s", builder_project["result"])

    logger.info("quota_agent redis_key: %s", builder_quota["redis_key"])
    logger.info("quota_agent result: %s", builder_quota["result"])


if __name__ == "__main__":
    main()
