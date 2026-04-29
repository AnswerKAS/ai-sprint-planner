import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

import yaml
from langchain_community.storage import RedisStore

from agents.agent_builder import agent_builder
from agents.agent_factory import init_inc_agent, init_project_agent, init_quota_agent, init_task_agent
from agents.critic_graph import MAX_ITERATIONS, build_critic_graph
from agents.tools.common_tools import TEAM_CAPACITY
from report.html_generator import IterationRecord, TeamReportData, generate_html_report
from agents.tools.critic_tools import fetch_candidates_context
from store.redis import REDIS_TTL, redis_client, save_critic_final_to_redis
from tasks.loader.excel_loader import load_tasks_from_excel
from tasks.model import SprintTask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _read_agent_configs() -> dict:
    with open("./agents/promts_agent.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _collect_report_data(
    team_name: str,
    session_id: str,
    task_list: list[SprintTask],
    critic_result: dict,
    agent_builder_results: dict[str, dict],
) -> TeamReportData:
    task_index = {t.task_id: t for t in task_list}
    capacity = TEAM_CAPACITY.get(team_name, 0.0)

    final_task_ids: list[str] = critic_result.get("selected_task_ids", [])
    final_tasks = [task_index[tid] for tid in final_task_ids if tid in task_index]
    total_sp = round(sum(t.sp for t in final_tasks), 2)
    utilization_pct = round(total_sp / capacity * 100, 1) if capacity else 0.0
    total_iterations: int = critic_result.get("iteration", 0)
    validated: bool = critic_result.get("validated", False)
    final_plan: str = critic_result.get("plan", "")

    # Specialist agent text outputs
    agent_results: dict[str, str] = {}
    for agent_key in ("inc_agent", "task_agent", "project_agent", "quota_agent"):
        builder_result = agent_builder_results.get(agent_key, {})
        result_payload = builder_result.get("result", {})
        messages = result_payload.get("messages", [])
        if messages:
            last = messages[-1]
            text = getattr(last, "content", "")
            agent_results[agent_key] = text if isinstance(text, str) else str(text)

    # Per-iteration data from Redis
    iterations: list[IterationRecord] = []
    for i in range(1, total_iterations + 1):
        iter_raw = redis_client.get(f"critic_agent:iteration:{i}:{session_id}")
        consult_raw = redis_client.get(f"critic_agent:consultation:{i}:{session_id}")

        iter_data = json.loads(iter_raw) if isinstance(iter_raw, str) else {}
        consult_data = json.loads(consult_raw) if isinstance(consult_raw, str) else {}

        iterations.append(IterationRecord(
            iteration=i,
            plan=iter_data.get("plan", ""),
            feedback=iter_data.get("feedback", ""),
            validated=iter_data.get("validated", False),
            total_sp=iter_data.get("total_sp", 0.0),
            consultation=consult_data.get("consultation", ""),
        ))

    return TeamReportData(
        team_name=team_name,
        session_id=session_id,
        capacity=capacity,
        final_task_ids=final_task_ids,
        final_tasks=final_tasks,
        total_sp=total_sp,
        utilization_pct=utilization_pct,
        total_iterations=total_iterations,
        validated=validated,
        final_plan=final_plan,
        agent_results=agent_results,
        iterations=iterations,
    )


async def main(team_name: str, task_list: list[SprintTask], configs: dict, store: RedisStore) -> TeamReportData:
    session_id = str(uuid.uuid4())
    logger.info("Session started: %s for team %s", session_id, team_name)

    logger.info("[%s] Running 4 specialist agents in parallel...", team_name)
    specialist_results = await asyncio.gather(
        agent_builder(
            agent=init_inc_agent,
            prompt=f"Какие инцидентные задачи взять в работу?\nteam_name = \"{team_name}\"",
            task_list=task_list,
            config=configs.get("inc_agent", {}),
            store=store,
            session_id=session_id,
            redis_client=redis_client,
            team_name=team_name,
            save_to_redis=True,
        ),
        agent_builder(
            agent=init_task_agent,
            prompt=f"Какие внутренние задачи взять в работу?\nteam_name = \"{team_name}\"",
            task_list=task_list,
            config=configs.get("task_agent", {}),
            store=store,
            session_id=session_id,
            redis_client=redis_client,
            team_name=team_name,
            save_to_redis=True,
        ),
        agent_builder(
            agent=init_project_agent,
            prompt=f"Какие проектные задачи взять в работу?\nteam_name = \"{team_name}\"",
            task_list=task_list,
            config=configs.get("project_agent", {}),
            store=store,
            session_id=session_id,
            redis_client=redis_client,
            team_name=team_name,
            save_to_redis=True,
        ),
        agent_builder(
            agent=init_quota_agent,
            prompt=f"Какие квотные задачи взять в работу?\nteam_name = \"{team_name}\"",
            task_list=task_list,
            config=configs.get("quota_agent", {}),
            store=store,
            session_id=session_id,
            redis_client=redis_client,
            team_name=team_name,
            save_to_redis=True,
        ),
    )

    agent_builder_results = {
        "inc_agent": specialist_results[0],
        "task_agent": specialist_results[1],
        "project_agent": specialist_results[2],
        "quota_agent": specialist_results[3],
    }

    # Fetch specialist candidates once — injected into every critic prompt directly,
    # so the critic agent never needs to call get_all_candidates per iteration.
    candidates_context = fetch_candidates_context(redis_client, session_id)

    logger.info("[%s] Running critic agent (reflection loop, up to %d iterations)...", team_name, MAX_ITERATIONS)
    critic_graph = build_critic_graph(
        config=configs.get("critic_agent", {}),
        task_list=task_list,
        store=store,
        redis_client=redis_client,
        session_id=session_id,
        team_name=team_name,
        candidates_context=candidates_context,
    )
    critic_result = await critic_graph.ainvoke({
        "iteration": 0,
        "plan": "",
        "selected_task_ids": [],
        "feedback": "",
        "validated": False,
        "consultation": "",
        "prev_remove_recs": [],
        "prev_add_recs": [],
    })

    final_plan = critic_result.get("plan", "")
    total_iterations = critic_result.get("iteration", 0)

    redis_key = save_critic_final_to_redis(
        redis_client=redis_client,
        session_id=session_id,
        team_name=team_name,
        plan=final_plan,
        total_iterations=total_iterations,
    )

    logger.info("critic_agent iterations used: %d", total_iterations)
    logger.info("critic_agent redis_key: %s", redis_key)
    logger.info("critic_agent final plan:\n%s", final_plan)
    if critic_result.get("feedback"):
        logger.warning("critic_agent note: %s", critic_result["feedback"])

    return _collect_report_data(
        team_name=team_name,
        session_id=session_id,
        task_list=task_list,
        critic_result=critic_result,
        agent_builder_results=agent_builder_results,
    )


async def _run_all(teams: list[str], task_list: list[SprintTask], configs: dict) -> None:
    redis_url = (
        f"redis://:{os.getenv('REDIS_PASSWORD', '')}@"
        f"{os.getenv('REDIS_HOST', 'localhost')}:"
        f"{os.getenv('REDIS_PORT', '6379')}/0"
    )
    store = RedisStore(redis_url=redis_url, ttl=REDIS_TTL)

    team_reports: list[TeamReportData] = await asyncio.gather(
        *[main(team_name=team, task_list=task_list, configs=configs, store=store) for team in teams]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = generate_html_report(list(team_reports), f"sprint_report_{timestamp}.html")
    logger.info("HTML report saved: %s", report_path)


if __name__ == "__main__":
    _configs = _read_agent_configs()
    _task_list = load_tasks_from_excel("sprint_tasks_template_short.xlsx")
    _teams = ["Python", "SA", "Meth"]

    asyncio.run(_run_all(_teams, _task_list, _configs))
