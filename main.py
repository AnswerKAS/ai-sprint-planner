from pydantic import BaseModel, Field
import yaml
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from agents.agents import ResponseFormat


### Чтение конфигурации AI агентов из YAML файла
def read_config_ai_agents() -> dict:
    with open("./agents/promts_agent.yaml", "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data


### Инициализация AI агентов на основе конфигурации

def init_inc_agent(config: dict) -> ChatOllama:
    agent_config = config["inc_agent"]
    agent = create_agent(
        model=ChatOllama(model="qwen3:4b-instruct", temperature=0),
        system_prompt=agent_config["sysmte_prompt"],
        tools=[],
        response_format=ResponseFormat
    )
    return agent

if __name__ == "__main__":
    ai_agents_configs = read_config_ai_agents()
    inc_agent = init_inc_agent(ai_agents_configs)
    result = inc_agent.invoke(
        {
    "messages": [
        {
            "role": "user",
            "content": """Привет, inc_agent! Какие инцидентные задачи ты бы выбрал для работы?
Задачи:
[
  {"task_id": 122, "text": "Не работает интернет в офисе"},
  {"task_id": 123, "text": "Не работает гарнитура"}
]"""
        }
    ]
}
     )
    
    print(result)