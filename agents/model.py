from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(description="Уникальный идентификатор задачи для каждой команды.")
    text: str = Field(description="Описание задачи")
    sp: float = Field(description="Story Points задачи")
    reasoning: str = Field(description="Обоснование выбора задач")


class ResponseAgent(BaseModel):
    team_name: str = Field(description="Название команды")
    task_list: list[Task] = Field(description="Список задач, которые выбрал агент, для команды")
    summary: str = Field(description="Краткое резюме по выбранным задачам. Возможно ли их взять в работу вместе?")