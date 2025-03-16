from abc import ABC, abstractmethod


class OpertaionType:
    def __init__(self, type_code=None):
        self.type_code = type_code


class BaseOperation(ABC):
    def __init__(self, color, duration, earliest_start, job_index):
        self.color = color
        self.index = None
        self.type = OpertaionType()
        self.duration = duration
        self.earliest_start = earliest_start
        self.job = None

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
