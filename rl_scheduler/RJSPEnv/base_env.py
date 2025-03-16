import json
from collections import defaultdict

import gymnasium as gym
import matplotlib.patches as mpatches  # 필요한 모듈을 가져옵니다.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space


class BaseEnv(gym.Env):
    def __init__(self, config, **kwargs):
        self.config = config
        self.action_space = None
        self.observation_space = None

    def reset():
        NotImplementedError()

    def step():
        NotImplementedError()

    def get_reward():
        NotImplementedError()

    def render():
        NotImplementedError()
