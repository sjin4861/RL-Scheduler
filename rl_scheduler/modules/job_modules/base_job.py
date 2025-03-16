from abc import ABC, abstractmethod

from rl_scheduler.modules.op_modules.base_op import BaseOperation


class BaseJob(ABC):
    def __init__(self, name, color, operations, index=None):
        self.name = name
        self.color = color
        self.operation_queue = [
            BaseOperation(
                op_info, job_id=str(int(name[4:]) - 1), color=color, job_index=index
            )
            for op_info in operations
        ]
        self.is_done = False

    @abstractmethod
    def get_job(self):
        pass
