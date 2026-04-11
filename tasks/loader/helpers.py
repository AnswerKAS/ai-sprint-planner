from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_str(value: Any) -> str:
    """
    Возвращает строку.
    Если значение пустое/NaN -> пустая строка.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_optional_str(value: Any) -> str | None:
    """
    Возвращает строку или None.
    Если значение пустое/NaN/"" -> None.
    """
    if pd.isna(value):
        return None

    value_str = str(value).strip()
    return value_str if value_str else None


def _safe_optional_float(value: Any) -> float | None:
    """
    Безопасно приводит значение к float | None.

    Поддерживает:
    - NaN -> None
    - bool -> 0.0 / 1.0
    - int/float -> float
    - строки вида "12", "12.5", "12,5"
    - "false"/"no"/"n" -> 0.0
    - "true"/"yes"/"y" -> 1.0
    - пустые значения -> None
    """
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip().lower()

    if value_str in ("", "none", "null", "nan"):
        return None

    if value_str in ("false", "no", "n", "нет"):
        return 0.0

    if value_str in ("true", "yes", "y", "да"):
        return 1.0

    value_str = value_str.replace(",", ".")

    try:
        return float(value_str)
    except ValueError:
        return None


def _safe_float(value: Any) -> float:
    """
    Безопасно приводит значение к float.
    Если преобразование невозможно -> 0.0
    """
    result = _safe_optional_float(value)
    return 0.0 if result is None else result


def _safe_int(value: Any) -> int:
    """
    Безопасно приводит значение к int.

    Поддерживает:
    - NaN -> 0
    - bool -> 0 / 1
    - int -> int
    - float -> int
    - строки вида "12", "12.0", "12,0"
    - "false"/"no"/"n" -> 0
    - "true"/"yes"/"y" -> 1
    """
    if pd.isna(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    value_str = str(value).strip().lower()

    if value_str in ("", "none", "null", "nan", "false", "no", "n", "нет"):
        return 0

    if value_str in ("true", "yes", "y", "да"):
        return 1

    try:
        return int(float(value_str.replace(",", ".")))
    except ValueError:
        return 0


def _safe_bool(value: Any) -> bool:
    """
    Безопасно приводит значение к bool.

    True для:
    - True
    - 1, ненулевых чисел
    - "true", "1", "yes", "y", "да"
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    value_str = str(value).strip().lower()
    return value_str in ("true", "1", "yes", "y", "да")


def _parse_escalation_texts(value: Any) -> list[str]:
    """
    Преобразует значение поля escalation_texts в list[str].

    Поддерживает:
    - NaN -> []
    - строку "text1; text2; text3" -> ["text1", "text2", "text3"]
    - строку с переносами строк
    - уже готовый список
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        # сначала нормализуем переносы строк в ;
        raw = raw.replace("\n", ";").replace("\r", ";")
        parts = [item.strip() for item in raw.split(";")]
        return [item for item in parts if item]

    return []