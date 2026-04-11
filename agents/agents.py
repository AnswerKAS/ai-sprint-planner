from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: int = Field(description="Уникальный идентификатор задачи")
    text: str = Field(description="Описание задачи")
    reasoning: str = Field(description="Обоснование выбора задач")

class ResponseFormat(BaseModel):
    task_list: list[Task] = Field(description="Список задач, которые выбрал агент")