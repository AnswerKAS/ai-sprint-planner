from langchain.agents import create_agent
from langchain_community.storage import RedisStore
from langchain_ollama import ChatOllama
from agents.tools.common_tools import get_team_capacity, get_team_members, make_get_inc_tasks_tool
from agents.model import ResponseAgent
from tasks.model import SprintTask


def init_inc_agent(config: dict, input_task_list: list[SprintTask], store: RedisStore) -> create_agent:
    agent_config = config["inc_agent"]
    agent = create_agent(
        model=ChatOllama(model="qwen3:4b-instruct", temperature=0),
        tools=[get_team_capacity, get_team_members,  make_get_inc_tasks_tool(input_task_list)],
        system_prompt=agent_config["system_prompt"] + "\n\n" + agent_config["limitations"],
        response_format=ResponseAgent,
        name="inc_agent",
        debug=False,
        store=store,
    )
    return agent