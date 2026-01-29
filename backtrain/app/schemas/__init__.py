from typing import Any
from pydantic import BaseModel


class TrainEntry(BaseModel):
    entry: dict[str, Any]
