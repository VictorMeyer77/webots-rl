from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.memory import Memory
from app.routers.environment import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.memory = Memory(capacity=100000)  # TODO: Move to config
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
