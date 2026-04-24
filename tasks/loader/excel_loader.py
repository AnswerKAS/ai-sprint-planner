import logging
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import ValidationError

from ..model import SprintTask
from .helpers import (
    _parse_escalation_texts,
    _safe_bool,
    _safe_float,
    _safe_int,
    _safe_optional_float,
    _safe_optional_str,
    _safe_str,
)

logger = logging.getLogger(__name__)


def load_tasks_from_excel(path: str) -> List[SprintTask]:
    full_path = Path(path).resolve()
    logger.info("Loading tasks from: %s", full_path)

    df = pd.read_excel(path)
    tasks: List[SprintTask] = []

    for idx, row in df.iterrows():
        try:
            task = SprintTask(
                task_id=str(row.get("task_id")),
                title=_safe_str(row.get("title")),
                user_story=_safe_optional_str(row.get("user_story")),
                team=_safe_str(row.get("team")),
                category=_safe_str(row.get("category")),
                priority=_safe_str(row.get("priority")),
                status=_safe_str(row.get("status")),
                stage=_safe_optional_str(row.get("stage")),
                customer_unit=_safe_optional_str(row.get("customer_unit")),
                sp=_safe_float(row.get("sp")),
                business_value=_safe_float(row.get("business_value")),
                has_escalation=_safe_bool(row.get("has_escalation")),
                escalation_count=_safe_int(row.get("escalation_count")),
                escalation_texts=_parse_escalation_texts(row.get("escalation_texts")),
                rice=_safe_optional_float(row.get("rice")),
                quota=_safe_optional_float(row.get("quota")),
            )
            tasks.append(task)
        except ValidationError as e:
            logger.warning("Skipping row %s due to validation error: %s", idx, e)

    logger.info("Loaded %d tasks", len(tasks))
    return tasks
