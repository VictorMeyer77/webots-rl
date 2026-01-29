from fastapi import APIRouter, Depends, HTTPException
from app.schemas import TrainEntry
from app.dependencies import get_memory
from app.core.memory import Memory

router = APIRouter(prefix="/environment", tags=["environment"])


@router.post("/{train_id}/{env_id}/{episode_id}/{step}")
def add_environment_step(
    train_id: str,
    env_id: int,
    episode_id: int,
    step: int,
    payload: TrainEntry,
    memory: Memory = Depends(get_memory),
):
    key = (train_id, env_id, episode_id, step)
    memory.add(key, payload)
    return {"status": "stored"}


@router.get("/{train_id}/{env_id}/{episode_id}/{step}")
def get_environment_step(
    train_id: str,
    env_id: int,
    episode_id: int,
    step: int,
    memory: Memory = Depends(get_memory),
):
    key = (train_id, env_id, episode_id, step)
    value = memory.get(key)

    if value is None:
        raise HTTPException(status_code=404, detail="Step not found")

    return value
