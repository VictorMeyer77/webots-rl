from fastapi import Request


def get_memory(request: Request):
    return request.app.state.memory
