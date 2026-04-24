from typing import Callable

from langchain.agents import create_agent
from langchain_community.storage import RedisStore
from langchain_ollama import ChatOllama

from agents.model import AgentContext, ResponseAgent
from agents.tools.common_tools import (
    get_team_capacity,
    get_team_members,
    make_get_inc_tasks_tool,
    make_get_task_tasks_tool,
    sum_sp_from_agent_response,
)
from tasks.model import SprintTask


def _create_sprint_agent(
    config: dict,
    input_task_list: list[SprintTask],
    store: RedisStore,
    task_tool_factory: Callable,
):
    sys_prompt = config["system_prompt"] + "\n\n" + config["limitations"]
    model_name = config.get("model", "qwen3:4b-instruct")
    agent_name = config.get("name", "agent")

    return create_agent(
        model=ChatOllama(model=model_name, temperature=0),
        tools=[
            get_team_capacity,
            get_team_members,
            sum_sp_from_agent_response,
            task_tool_factory(input_task_list),
        ],
        system_prompt=sys_prompt,
        response_format=ResponseAgent,
        name=agent_name,
        debug=False,
        store=store,
        context_schema=AgentContext,
    )


def init_inc_agent(config: dict, input_task_list: list[SprintTask], store: RedisStore):
    return _create_sprint_agent(config, input_task_list, store, make_get_inc_tasks_tool)


def init_task_agent(config: dict, input_task_list: list[SprintTask], store: RedisStore):
    return _create_sprint_agent(config, input_task_list, store, make_get_task_tasks_tool)
