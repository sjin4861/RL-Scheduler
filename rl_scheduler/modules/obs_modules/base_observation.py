from abc import ABC, abstractmethod


class BaseObservation(ABC):
    @abstractmethod
    def get_observation(self, *args, **kwargs):
        pass
