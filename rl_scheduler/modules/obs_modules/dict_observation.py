from gymnasium import spaces

from rl_scheduler.modules.obs.base_observation import BaseObservation


class DictObservation(BaseObservation):
    def __init__(self, config):
        super().__init__(config)
        self.observation_space = spaces.Dict(config["observation_space"])

    def get_observation(self, **kwargs):
        return self.observation_space.sample()

    def get_observation_space(self):
        return self.observation_space
