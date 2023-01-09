from typing import Generic, TypeVar

DT = TypeVar("DT")


class Message:
    pass


class PartialResultMessage(Message, Generic[DT]):
    def __init__(self, rank: int, data: DT):
        self.rank = rank
        self.data = data


class DoneMessage(Message):
    def __init__(self, rank: int):
        self.rank = rank