from pydantic import BaseModel, Field

from typing import List, Optional
from pydantic import BaseModel, Field


class SprintTask(BaseModel):
    task_id: str = Field(..., description="Уникальный идентификатор задачи")

    title: str = Field(..., description="Название задачи")
    user_story: Optional[str] = Field(None, description="User story")

    team: str = Field(..., description="Команда, ответственная за задачу")

    category: str = Field(
        ...,
        description="Тип задачи (project / incident / support / internal)"
    )

    priority: str = Field(
        ...,
        description="Приоритет (low / medium / high / critical)"
    )

    status: str = Field(
        ...,
        description="Статус (new / ready / in_progress / done и т.д.)"
    )

    stage: Optional[str] = Field(
        None,
        description="Стадия (например discovery / development / testing)"
    )

    customer_unit: Optional[str] = Field(
        None,
        description="Подразделение заказчика"
    )

    sp: float = Field(
        ...,
        ge=0,
        description="Story Points"
    )

    business_value: float = Field(
        ...,
        ge=0,
        description="Бизнес-ценность"
    )

    has_escalation: bool = Field(
        default=False,
        description="Есть ли эскалация"
    )

    escalation_count: int = Field(
        default=0,
        ge=0,
        description="Количество эскалаций"
    )

    escalation_texts: List[str] = Field(
        default_factory=list,
        description="Тексты эскалаций"
    )

    rice: Optional[float] = Field(
        None,
        ge=0,
        description="RICE score"
    )

    quota: Optional[float] = Field(
        None,
        ge=0,
        description="Квота/лимит (если используется)"
    )