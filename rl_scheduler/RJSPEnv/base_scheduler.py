import copy
import heapq
import io
from abc import abstractmethod

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from stable_baselines3.common.env_checker import check_env


class BaseScheduler:
    def __init__(self, config):
        self.config = config
        self.jobs = None
        self.operations = None
        self.job_colors = None
        self.job_colors_dict

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_reward(self):
        pass

    @abstractmethod
    def render(self):
        pass
