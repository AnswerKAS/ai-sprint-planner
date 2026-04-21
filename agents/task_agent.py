from langchain.agents import create_agent
from langchain_community.storage import RedisStore
from langchain_ollama import ChatOllama
from agents.tools.common_tools import get_team_capacity, get_team_members, make_get_inc_tasks_tool, make_get_task_tasks_tool, sum_sp_from_agent_response
from agents.model import AgentContext, ResponseAgent
from tasks.model import SprintTask


def init_task_agent(config: dict, input_task_list: list[SprintTask], store: RedisStore) -> create_agent:

    sys_prompt = config["system_prompt"] + "\n\n" + config["limitations"]

    agent = create_agent(
        model=ChatOllama(model="qwen3:4b-instruct", temperature=0),
        tools=[get_team_capacity, get_team_members, sum_sp_from_agent_response, make_get_task_tasks_tool(input_task_list) ],
        system_prompt=sys_prompt,
        response_format=ResponseAgent,
        name="task_agent",
        debug=False,
        store=store,
        context_schema=AgentContext
    )
    return agent