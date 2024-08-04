import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from stable_baselines3.common.env_checker import check_env
from scheduler_env.customScheduler_repeat import customRepeatableScheduler

class SchedulingEnv(gym.Env):
    def _load_machines(self, file_path):
        machines = []

        with open(file_path, 'r') as file:
            data = json.load(file)

        for machine_data in data["machines"]:
            machine = {}
            machine['name'] = machine_data["name"]
            machine['ability'] = machine_data["type"].split(', ')
            machines.append(machine)

        return machines

    def _load_jobs_repeat(self, file):
        # Just in case we are reloading operations

        jobs = []  # 리턴할 용도
        jobs_new_version = []  # 파일 읽고 저장할 때 쓰는 용도
        f = open(file)

        # returns JSON object as  a dictionary
        data = json.load(f)
        f.close()
        jobs_new_version = data['jobs']

        for job in jobs_new_version:
            job_info = {}
            # Initial index of steps within job
            job_info['name'] = job['name']
            job_info['color'] = job['color']
            # changed : add deadline
            job_info['deadline'] = job['deadline']
            earliestStart = job['earliest_start']

            operations = []
            for operation in job['operations']:
                predecessor = operation['predecessor']
                operation_info = {}
                # Sequence is the scheduling job, the series of which defines a State or Node.
                operation_info['sequence'] = None
                operation_info['index'] = operation['index']
                operation_info['type'] = operation['type']
                if predecessor is None:
                    operation_info['predecessor'] = None
                    operation_info['earliest_start'] = earliestStart
                else:
                    operation_info['predecessor'] = predecessor
                    operation_info['earliest_start'] = None
                operation_info['duration'] = operation['duration']
                operation_info['start'] = None
                operation_info['finish'] = None

                operations.append(operation_info)

            job_info['operations'] = operations
            jobs.append(job_info)

        return jobs

    def __init__(self, machine_config_path, job_config_path, job_repeats_params, render_mode="seaborn", weight_final_time=80, weight_job_deadline=20, weight_op_rate=0, target_time = None, test_mode=False, max_time = 150):
        super(SchedulingEnv, self).__init__()
        self.weight_final_time = weight_final_time
        self.weight_job_deadline = weight_job_deadline
        self.weight_op_rate = weight_op_rate
        self.target_time = target_time
        self.total_durations = 0
        self.job_repeats_params = job_repeats_params  # 각 Job의 반복 횟수에 대한 평균과 표준편차
        self.current_repeats = [job_repeat[0] for job_repeat in job_repeats_params]
        self.test_mode = test_mode
        self.best_makespan = float('inf')  # 최적 makespan

        self.jobs = self._load_jobs_repeat(job_config_path)
        self.machine_config = self._load_machines(machine_config_path)

        self.custom_scheduler = None

        self.len_machines = len(self.machine_config)
        self.len_jobs = len(self.jobs)

        self.num_steps = 0

        self.action_space = spaces.Discrete(self.len_machines * self.len_jobs)
        self.observation_space = spaces.Dict({
            "action_masks": spaces.Box(low=0, high=1, shape=(self.len_machines * self.len_jobs, ), dtype=np.int8),
            "job_details": spaces.Box(low=-1, high=25, shape=(len(self.jobs), 4, 2), dtype=np.int8),
            #'machine_operation_rate': spaces.Box(low=0, high=1, shape=(self.len_machines, ), dtype=np.float32),
            #"machine_types": spaces.Box(low=0, high=1, shape=(self.len_machines, 25), dtype=np.int8),
            "schedule_heatmap": spaces.Box(low=0, high=1, shape=(self.len_machines, max_time), dtype=np.int8),
            ### 아래는 render 함수의 결과를 배열로 전달하는 것
            #"schedule_image" : spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),  # 이미지 공간 설정
            #"schedule_buffer": spaces.Box(low=-1, high=15, shape=(self.len_jobs, 2), dtype=np.int64),
            "schedule_buffer_job_repeat": spaces.Box(low=-1, high=10, shape=(self.len_jobs, ), dtype=np.int64),
            "schedule_buffer_operation_index": spaces.Box(low=-1, high=10, shape=(self.len_jobs, ), dtype=np.int64),
            "estimated_tardiness": spaces.Box(low=-1, high=10, shape=(self.len_jobs, ), dtype=np.float64),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._initialize_scheduler()
        self.num_steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Map the action to the corresponding machine and job
        selected_machine_id = action // self.len_jobs
        selected_job_id = action % self.len_jobs
        action = [selected_machine_id, selected_job_id]

        # error_action이 아니라면 step의 수를 증가시킨다
        self.num_steps += 1
        reward = 0.0

        if self._is_legal(action):
            self._update_state(action)
            reward += self._calculate_step_reward()
        else:  # Illegal action
            reward = -0.5

        terminated = self._is_done()
        if terminated:
            final_makespan = self.custom_scheduler._get_final_operation_finish()
            self.best_makespan = min(self.best_makespan, final_makespan)  # Update the best makespan
            reward = self._calculate_final_reward()
            self.custom_scheduler.cal_final_cost()

        truncated = bool(self.num_steps == 10000)
        if truncated:
            reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _is_legal(self, action):
        return self.custom_scheduler.is_legal(action)

    def _is_done(self):
        return self.custom_scheduler.is_done()

    def _update_state(self, action):
        self.custom_scheduler.update_state(action)

    def _get_observation(self):
        observation = self.custom_scheduler.get_observation()
        #observation["schedule_image"] = self.custom_scheduler.render_image_to_array(num_steps = self.num_steps)[:, :, :3]
        return observation
    
    def set_test_mode(self):
        self.test_mode = True
        self.reset()

    # For MaskablePPO
    def action_masks(self):
        return self.custom_scheduler.action_masks()

    def _get_info(self):
        info = self.custom_scheduler.get_info()
        info['num_steps'] = self.num_steps
        info['current_repeats'] = self.current_repeats
        return info

    def _calculate_final_reward(self):
        return self.custom_scheduler.calculate_final_reward(total_durations=self.total_durations, weight_final_time=self.weight_final_time, weight_job_deadline=self.weight_job_deadline, weight_op_rate=self.weight_op_rate, target_time=self.target_time)

    def _calculate_step_reward(self):
        return self.custom_scheduler.calculate_step_reward()

    def _initialize_scheduler(self):
        if self.test_mode:
            repeats_list = self.current_repeats[::]
        # 각 Job의 반복 횟수를 랜덤하게 설정
        # 랜덤 반복 횟수에 따라 Job 인스턴스를 생성
        else:
            repeats_list = []
            for mean, std in self.job_repeats_params:
                repeats = max(1, int(np.random.normal(mean, std)))
                repeats_list.append(repeats)
            self.current_repeats = repeats_list[::]
            
        self._calculate_target_time()

        random_jobs = []
        for job, repeat in zip(self.jobs, repeats_list):
            random_job_info = {
                'name': job['name'],
                'color': job['color'],
                'operations': job['operations'],
                'deadline': job['deadline'][:repeat]  # 주어진 반복 횟수에 따라 deadline 설정
            }
            random_jobs.append(random_job_info)

        # 랜덤 Job 인스턴스를 사용하여 customScheduler 초기화
        self.custom_scheduler = customRepeatableScheduler(jobs=random_jobs, machines=self.machine_config)
        self.custom_scheduler.reset()

    def _calculate_target_time(self):
        total_duration = 0
        for i in range(len(self.jobs)):
            job_duration = sum(op['duration'] for op in self.jobs[i]['operations'])
            total_duration += job_duration * self.current_repeats[i]
        
        self.total_durations = total_duration
        self.target_time = total_duration / self.len_machines

    def render(self, mode="human"):
        self.custom_scheduler.render(mode=mode, num_steps=self.num_steps)

    def visualize_graph(self):
        self.custom_scheduler.visualize_graph()


    def print_result(self, info, detail_mode = False):
        current_repeats = info['current_repeats']
        print(f"Current Repeats\t\t\t:\t{current_repeats}")

        # 최종 점수 출력
        reward = info["reward"]
        print(f"Goal reached! Final score\t:\t{reward:.2f}")

        cost_deadline = info["cost_deadline"]
        cost_hole = info["cost_hole"]
        cost_processing = info["cost_processing"]
        cost_makespan = info["cost_makespan"]
        sum_costs = cost_deadline + cost_hole + cost_processing + cost_makespan
        profit = env.total_durations / 100 * 10 - sum_costs
        print(f"Total revenue = {profit:.2f} - {sum_costs:.2f} = {profit - sum_costs:.2f}")
        print(f"Sum of Costs\t\t\t:\t{cost_deadline + cost_hole + cost_processing + cost_makespan:.2f}")
        print(f"Cost Deadline\t\t\t:\t{cost_deadline:.2f}")
        print(f"Cost Hole\t\t\t:\t{cost_hole:.2f}")
        print(f"Cost Processing\t\t:\t{cost_processing:.2f}")
        print(f"Cost Makespan\t\t\t:\t{cost_makespan:.2f}")


        # 최종 완료 시간 출력
        env = info["env"]
        print(f"Finish Time / Target Time\t:\t{info['finish_time']} / {int(env.target_time)}")

        # jobs 생성
        jobs = []
        job_deadlines = info['job_deadline']
        job_tardiness = info['job_time_exceeded']
        index = 0

        if detail_mode:
            for repeat in current_repeats:
                for r in range(repeat):
                    deadline = job_deadlines[index+r]
                    tardiness = job_tardiness[index+r]
                    print(f"Job {index + 1} - Repeat {r + 1}\t\t:\tTardiness/Deadline = {tardiness}/{deadline}")
                index += 1
            
        for job_id, repeat in enumerate(current_repeats, 1):
            job_info = {
                'job_id': job_id,
                'tardiness': job_tardiness[index:index+repeat],
                'deadline': job_deadlines[index:index+repeat]
            }
            jobs.append(job_info)
            index += repeat

        
        # Calculate Tardiness/Deadline ratios and assign colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78']  # Color palette for jobs
        
        ratios = []
        x_labels = []
        bar_colors = []
        x_positions = []
        
        current_x = 0
        for job in jobs:
            ratio = [t/d if d != 0 else 0 for t, d in zip(job['tardiness'], job['deadline'])]
            ratios.extend(ratio)
            x_labels.extend([f'Job {job["job_id"]} - Repeat {i+1}' for i in range(len(ratio))])
            bar_colors.extend([colors[job['job_id'] - 1]] * len(ratio))
            x_positions.extend([current_x + i for i in range(len(ratio))])
            current_x += len(ratio) + 1  # Add space between different jobs

        # Calculate and print the average Tardiness/Deadline ratio
        avg_ratio = np.mean(ratios)
        print(f"Average Tardiness/Deadline Ratio:\t{avg_ratio:.2f}")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.8
        ax.bar(x_positions, ratios, width=bar_width, color=bar_colors)
        
        # Set x-ticks and x-tick labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90, ha='center')
        ax.set_xlabel('Job - Repeat')
        ax.set_ylabel('Tardiness/Deadline Ratio')
        ax.set_title('Tardiness/Deadline Ratio per Job Repeat')
        
        # Legend
        unique_jobs = list(set([f'Job {job["job_id"]}' for job in jobs]))
        legend_patches = [mpatches.Patch(color=colors[i], label=unique_jobs[i]) for i in range(len(unique_jobs))]
        legend_patches.sort(key=lambda x: int(x.get_label().split()[1]))
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env_8_12_1_t = SchedulingEnv(machine_config_path= "instances/Machines/v0-8.json", job_config_path = "instances/Jobs/v0-12-repeat.json", job_repeats_params = [(1, 1)] * 12, test_mode = True)
    env = env_8_12_1_t
    check_env(env)

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
