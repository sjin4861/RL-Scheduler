import gymnasium as gym
from gymnasium import spaces
import json
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from stable_baselines3.common.env_checker import check_env


class Resource():
    def __init__(self, resouces_dictionary):
        self.task_schedule = [] # (tasks)
        self.name = resouces_dictionary['name'] 
        self.ability = resouces_dictionary['ability'] # "A, B, C, ..."
        self.reward = 0

    def __str__(self):
        # str_to_tasks = [str(task) for task in self.task_schedule]
        # return f"{self.name} : {str_to_tasks}"
        return f"{self.name}"


class Order():
    def __init__(self, order_dictionary):
        self.name = order_dictionary['name']
        self.color = order_dictionary['color']
        self.task_queue = [Task(task_dictionary) for task_dictionary in order_dictionary['tasks']]
        self.reward = 0

class Task():
    def __init__(self, task_dictionary):
        self.sequence = task_dictionary['sequence']
        self.step = task_dictionary['step']
        self.type = task_dictionary['type']
        self.predecessor = task_dictionary['predecessor']
        self.earliest_start = task_dictionary['earliest_start']
        self.duration = task_dictionary['duration']
        self.start = task_dictionary['start']
        self.finish = task_dictionary['finish']
        self.resource = -1
        self.color = ""
        self.order = -1

    def to_dict(self):
        return {
            'sequence': self.sequence,
            'step' : self.step,
            'type' : self.type,
            'predecessor' : self.predecessor,
            'earliest_start' : self.earliest_start,
            'duration' : self.duration,
            'start': self.start,
            'finish': self.finish,
            'resource': self.resource,
            'color' : self.color,
            'order' : self.order
        }
    def __str__(self):
        return f"order : {self.order}, step : {self.step} | ({self.start}, {self.finish})"

class SchedulingEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    def load_resources(self, file_path):
        resources = []

        with open(file_path, 'r') as file:
            data = json.load(file)

        for resource_data in data["resources"]:
            resource = {}
            resource['name'] = resource_data["name"]
            resource['ability'] = resource_data["type"].split(', ')
            resources.append(resource)

        return resources
    
    def load_orders_new_version(self, file):
        # Just in case we are reloading tasks
        
        orders = [] # 리턴할 용도
        orders_new_version = [] # 파일 읽고 저장할 때 쓰는 용도
        f = open(file)

        # returns JSON object as  a dictionary
        data = json.load(f)
        f.close()
        orders_new_version = data['orders']

        for order in orders_new_version:
            order_dictonary = {}
            # Initial index of steps within order
            order_dictonary['name'] = order['name']
            order_dictonary['color'] = order['color']
            earliestStart = order['earliest_start']

            tasks = []
            for task in order['tasks']:
                predecessor = task['predecessor']
                task_dictionary = {}
                # Sequence is the scheduling order, the series of which defines a State or Node.
                task_dictionary['sequence'] = None
                task_dictionary['step'] = task['step']
                task_dictionary['type'] = task['type']
                if predecessor is None:
                    task_dictionary['predecessor'] = None
                    task_dictionary['earliest_start'] = earliestStart
                else:
                    task_dictionary['predecessor'] = predecessor
                    task_dictionary['earliest_start'] = None
                task_dictionary['duration'] = task['duration']
                task_dictionary['start'] = None
                task_dictionary['finish'] = None

                tasks.append(task_dictionary)
            
            order_dictonary['tasks'] = tasks
            orders.append(order_dictonary)

        return orders

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render.modes": ["seaborn"]}
    #resources_json, orders_json,
    def __init__(self, resources = "../resources/resources-default.json", orders = "../orders/orders-new-version.json", render_mode="seaborn"):
        super(SchedulingEnv, self).__init__()

        resources = self.load_resources(resources)
        orders = self.load_orders_new_version(orders)
        # Find the maximum 'resource' and 'predecessor' values in the tasks list
        self.resources = [Resource(resource_info) for resource_info in resources]
        self.orders = [Order(order_info) for order_info in orders]

        self.original_orders = copy.deepcopy(self.orders)
        self.original_resources = copy.deepcopy(self.resources)
        # 각 오더의 번호와 매칭되는 버퍼, 추후 크기를 키울 예정
        self.schedule_buffer = [-1 for _ in range(len(self.orders))]
        len_resource = len(self.resources)
        len_orders = len(self.orders)

        #추후 수정 
        max_predecessor = 10 
        self.original_tasks = [order.task_queue for order in self.orders]
        self.num_tasks = sum([len(order.task_queue) for order in self.orders])
        self.invalid_count = 0
        max_tasks = max([len(order.task_queue) for order in self.orders])

        self.action_space = spaces.MultiDiscrete([len_resource, len_orders])
        self.observation_space = spaces.Dict({
            'resource_reward': spaces.Box(low=0, high=5000, shape=(len_resource,), dtype=np.int32),
            'order_reward' : spaces.Box(low=0, high=5000, shape=(len_orders,), dtype=np.int32),
            'schedule_buffer' : spaces.Box(low=-1, high=max_tasks, shape=(len_orders,),dtype=np.int32),
            'duration': spaces.Box(low=0, high=5000, shape=(len_orders,), dtype=np.int32),
            'start': spaces.Box(low=-1, high=5000, shape=(len_orders,), dtype=np.int32),
            'finish': spaces.Box(low=-1, high=5000, shape=(len_orders,), dtype=np.int32),
        })
        
        self.current_task_info = copy.deepcopy([order.task_queue for order in self.orders])
        self.current_schedule = []
        self.num_scheduled_tasks = 0
        self.num_steps = 0
        self.finish_time = 0

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.current_schedule = []
        self.orders = copy.deepcopy(self.original_orders)
        self.resources = copy.deepcopy(self.original_resources)
        self.schedule_buffer = [-1 for _ in range(len(self.orders))]
        self.invalid_count = 0
        self.finish_time = 0
        # for i in range(len(self.orders)):
        #     self.orders[i].task_queue = self.current_task_info[i]   
        #     self.orders[i].reward = 0
            
        self.num_scheduled_tasks = 0
        self.num_steps = 0

        # for i in range(len(self.resources)):
        #     self.resources[i].reward = 0
        #     self.resources[i].task_schedule = []

        return self._get_observation(), {}  # empty info dict

    def step(self, action):
        def error_action(act):
            return act[0] < 0 or act[1] < 0 or act[0] >= len(self.resources) or act[1] >= len(self.orders)
        if error_action(action):
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        self._possible_schedule_list()
        self.num_steps += 1

        invalid_action = False

        if self.schedule_buffer[action[1]] < 0:
            invalid_action = True

        if not invalid_action:
            self._schedule_task(action)
            self._calculate_step_reward(action)
            reward = 2
        else:
            self.invalid_count += 1
            reward = -5

        terminated = bool(self.schedule_buffer.count(-1) == len(self.orders))

        if terminated:
            self.finish_time = self._get_final_task_finish()
            reward = self.finish_time

        truncated = bool(self.num_steps == 1000)

        # Optionally we can pass additional info, we are not using that for now
        info = {
            'finish_time' : self.finish_time,
            'invalid_count' : self.invalid_count
               }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self, mode="seaborn"):
        if mode == "console":
            # You can implement console rendering if needed
            pass
        elif mode == "seaborn":
            return self._render_seaborn()
        elif mode == "rgb_array":
            return self._render_rgb_array()

    def _render_seaborn(self):
        fig = self._make_chart()
        plt.show()

    def _render_rgb_array(self):
        # Render the figure as an image
        fig = self._make_chart()
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()

        # Convert the image to RGB array
        buf = canvas.buffer_rgba()
        width, height = canvas.get_width_height()
        rgb_array = np.frombuffer(
            buf, dtype=np.uint8).reshape((height, width, 4))

        return rgb_array

    def _make_chart(self):
        # Create a DataFrame to store task scheduling information
        current_schedule = [task.to_dict() for task in self.current_schedule]
        
        scheduled_df = list(
            filter(lambda task: task['sequence'] is not None, current_schedule))
        scheduled_df = pd.DataFrame(scheduled_df)

        if scheduled_df.empty:
            # Create an empty chart
            plt.figure(figsize=(12, 6))
            plt.title("Task Schedule Visualization")
            return plt

        # Create a bar plot using matplotlib directly
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(len(self.resources)):
            resource_tasks = scheduled_df[scheduled_df['resource'] == i]

            # Discriminate rows by lines
            line_offset = i - 0.9  # Adjust the line offset for better visibility

            for index, task in resource_tasks.iterrows():
                ax.bar(
                    # Adjust 'x' to start from 'start'
                    x=task["start"] + task["duration"] / 2,
                    height=0.8,  # Height of the bar
                    width=task["duration"],  # Width of the bar
                    bottom=line_offset,  # Discriminate rows by lines
                    color=task['color'],
                    alpha=0.7,  # Transparency
                    label=f'Task {int(task["step"])}',  # Label for the legend
                )

        # Set y-axis ticks to show every resource
        ax.set_yticks(np.arange(0, len(self.resources)))
        ax.set_yticklabels(self.resources)

        ax.set(ylabel="Resource", xlabel="Time")
        # Place the legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("Task Schedule Visualization")
        # 경고 무시 설정
        plt.rcParams['figure.max_open_warning'] = 0

        return fig

    def close(self):
        pass

    def _possible_schedule_list(self, target_order = None):
        # target_order은 매번 모든 Order를 보는 계산량을 줄이기 위해 설정할 변수
        # None은 최초의 호출에서, 또는 Reset이 이뤄질 경우를 위해 존재
        if target_order == None:
            buffer_index = 0
            
            for order in self.orders:
                # Assume order['steps'] is a list of tasks for the current order
                
                selected_task_index = -1
                
                for i in range(len(order.task_queue)):
                    # 아직 스케줄링을 시작하지 않은 Task를 찾는다
                    if order.task_queue[i].finish is None:
                        selected_task_index = i
                        break
                # 스케줄링 하지 않은 Task를 발견했다면        
                if selected_task_index >= 0:
                    selected_task = order.task_queue[selected_task_index]
        
                    # 만약 초기 시작 제한이 없다면 
                    # 초기 시작 제한을 이전 Task의 Finish Time으로 걸어주고 버퍼에 등록한다.
                    if selected_task.earliest_start is None:
                        if selected_task_index > 0:
                            selected_task.earliest_start = order.task_queue[selected_task_index-1].finish
                
                self.schedule_buffer[buffer_index] = selected_task_index
                buffer_index += 1
                
        # Action으로 인해 봐야할 버퍼의 인덱스가 정해짐

        # else:
        #     selected_task_index = -1
        #     for i in range(len(self.orders[target_order].task_queue)):
        #         # 아직 스케줄링을 시작하지 않은 Task를 찾는다
        #         if order.task_queue[i].finish is None:
        #             selected_task_index = i
        #             break
        #     if selected_task_index >= 0:
        #         selected_task = order.task_queue[selected_task_index]

        #         # 만약 초기 시작 제한이 없다면 
        #         # 초기 시작 제한을 이전 Task의 Finish Time으로 걸어주고 버퍼에 등록한다.
        #         if selected_task.earliest_start is None:
        #             if selected_task_index > 0:
        #                 selected_task.earliest_start = order.task_queue[selected_task_index-1].finish
            
        #     self.schedule_buffer[target_order] = selected_task_index        
        
    def _schedule_task(self, action):
        # Implement the scheduling logic based on the action
        # You need to update the start and finish times of the tasks
        # based on the selected task index (action) and the current state.

        # Example: updating start and finish times
        selected_resource = self.resources[action[0]]
        selected_order = self.orders[action[1]]
        selected_task = selected_order.task_queue[self.schedule_buffer[action[1]]]
        task_earliest_start = selected_task.earliest_start
        task_index = selected_task.step
        task_duration = selected_task.duration
        resource_tasks = sorted(selected_resource.task_schedule, key=lambda task: task.start)

        open_windows = []
        start_window = 0
        last_alloc = 0

        for scheduled_task in resource_tasks:
            resource_init = scheduled_task.start

            if resource_init > start_window:
                open_windows.append([start_window, resource_init])
            start_window = scheduled_task.finish

            last_alloc = max(last_alloc, start_window)

        # Fit the task within the first possible window
        window_found = False
        if task_earliest_start is None:
            task_earliest_start = 0

        for window in open_windows:
            # Task could start before the open window closes
            if task_earliest_start <= window[1]:
                # Now let's see if it fits there
                potential_start = max(task_earliest_start, window[0])
                if potential_start + task_duration <= window[1]:
                    # Task fits into the window
                    min_earliest_start = potential_start
                    window_found = True
                    break

        # If no window was found, schedule it after the end of the last task on the resource
        if not window_found:
            if task_earliest_start > 0:
                min_earliest_start = max(task_earliest_start, last_alloc)
            else:
                min_earliest_start = last_alloc

        # schedule it
        selected_task.sequence = self.num_scheduled_tasks + 1
        selected_task.start = min_earliest_start
        selected_task.finish = min_earliest_start + task_duration
        selected_task.resource = action[0]

        # 사실 여기서 color랑 order를 주는건 적절치 않은 코드임!!!!
        selected_task.color = self.orders[action[1]].color
        selected_task.order = action[1]

        self.current_schedule.append(selected_task)
        selected_resource.task_schedule.append(selected_task)
        self.num_scheduled_tasks += 1
        return

    def _get_final_task_finish(self):
        # Implement your reward function based on the current state.
        # You can use the start and finish times of tasks to calculate rewards.
        # Example: reward based on minimizing the makespan
        makespan = max(self.current_schedule,
                       key=lambda x: x.finish).finish
        
        #sum_of_orders_reward = sum([order.reward for order in self.orders])
        #sum_of_resources_reward = sum([resource.reward for resource in self.resources])
        
        return -makespan # + sum_of_orders_reward + sum_of_resources_reward # Negative makespan to convert it into a minimization problem

    def _calculate_step_reward(self, action):
        # 이 부분의 Reward 체계화 필요
        self.resources[action[0]].reward += np.log(self.resources[action[0]].reward + 10)
        self.orders[action[1]].reward += np.log(self.orders[action[1]].reward + 10)

    def _get_observation(self):
        observation = {
            'resource_reward' : np.array([resource.reward for resource in self.resources], dtype=np.int32),
            'order_reward' : np.array([order.reward for order in self.orders], dtype=np.int32),
            'schedule_buffer' : np.array(self.schedule_buffer, dtype=np.int32),
            'duration': np.array([self.orders[order_index].task_queue[task_index].duration if task_index >= 0 else 0 for order_index, task_index in enumerate(self.schedule_buffer)], dtype=np.int32),
            'start': np.array([self.orders[order_index].task_queue[task_index].start if task_index >= 0 and self.orders[order_index].task_queue[task_index].start is not None else -1 for order_index, task_index in enumerate(self.schedule_buffer)], dtype=np.int32),
            'finish': np.array([self.orders[order_index].task_queue[task_index].finish if task_index >= 0 and self.orders[order_index].task_queue[task_index].finish is not None else -1 for order_index, task_index in enumerate(self.schedule_buffer)], dtype=np.int32),
        }
        return observation

if __name__ == "__main__":
    env = SchedulingEnv()

    step = 0
    obs, _ = env.reset()

    while True:
        step += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Goal reached!", "reward=", reward)
            env.render()
            break