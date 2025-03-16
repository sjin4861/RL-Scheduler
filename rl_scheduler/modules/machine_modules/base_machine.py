from abc import ABC, abstractmethod


class BaseMachine(ABC):
    def __init__(self, machines_dictionary):
        self.operation_schedule = []  # (operations)
        self.name = None
        self.ability = None

    @abstractmethod
    def __str__(self):
        pass
