from collections import OrderedDict
from app.schemas import TrainEntry

Key = tuple[str, int, int, int]


class Memory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = OrderedDict()

    def add(self, key: Key, value: TrainEntry) -> None:
        self.memory[key] = value
        if len(self.memory) > self.capacity:
            self.memory.popitem(last=False)

    def get(self, key: Key) -> TrainEntry | None:
        return self.memory.get(key, None)

    def __contains__(self, key: Key) -> bool:
        return key in self.memory

    def __len__(self) -> int:
        return len(self.memory)
