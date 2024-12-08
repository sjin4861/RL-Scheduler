import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from stable_baselines3.common.env_checker import check_env
import matplotlib.colors as mcolors
import seaborn as sns # type: ignore
import heapq
from PIL import Image
import io

def type_encoding(type):
    type_code = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
                 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
    return type_code[type]

class Machine():
    def __init__(self, machines_dictionary):
        self.operation_schedule = []  # (operations)
        self.name = machines_dictionary['name']
        self.ability = self.ability_encoding(
            machines_dictionary['ability'])  # "A, B, C, ..."
        self.operation_rate = 0.0

    def __str__(self):
        # str_to_operations = [str(operation) for operation in self.operation_schedule]
        # return f"{self.name} : {str_to_operations}"
        return f"name : {self.name}, ability : {self.ability}"

    def ability_encoding(self, ability):
        return [type_encoding(type) for type in ability]

    def cal_last_finish_time(self):
        if self.operation_schedule:
            return max([op.finish for op in self.operation_schedule])
        else:
            return 0
        
    def cal_idle_time(self):
        # 선택된 machine에 idle time이 있는지 확인
        if not self.operation_schedule:
            return 0
        self.operation_schedule.sort(key=lambda x: x.start)
        first_start = self.operation_schedule[0].start
        last_finish = self.operation_schedule[-1].finish
        total_duration = sum(op.duration for op in self.operation_schedule)
        hole_time = last_finish - first_start - total_duration
        return hole_time
    
    def encode_ability(self):
        # 각 머신의 능력을 encoding 한다
        # [0, 1, 3]이 ablity라고 치면 해당 머신이 0, 1, 3 type의 operation을 처리할 수 있다는 것 이므로
        # 이를 [True, True, False, True, False, ...]로 간주하고
        # 1011 -> 7로 encoding한다

        ability = 0
        for i in self.ability:
            ability += 2**i
        return ability

    def can_process_operation(self, operation_type):
        return operation_type in self.ability
    
    # 코드 검증 필요
    def cal_best_finish_time(self, op_duration, op_type, op_earliest_start):
        if not self.can_process_operation(op_type):
            return -1
        
        if not self.operation_schedule:
            return op_earliest_start + op_duration
        
        operation_schedule = self.operation_schedule[::]
        operation_schedule.sort(key=lambda x: x.start)

        best_start_time = 0
        if len(operation_schedule) == 1:
            if operation_schedule[0].start >= op_earliest_start + op_duration:
                best_start_time = op_earliest_start
            else:
                best_start_time = max(operation_schedule[0].finish, op_earliest_start)
            return best_start_time + op_duration
        else:
            for i in range(1, len(operation_schedule)):
                # 이러면 고민할 게 없어진다. 이후 hole을 찾고 들어갈 수 있는지 판단한다.
                if op_earliest_start <= operation_schedule[i-1].finish:
                    #아래는 i-1과 i 사이에 들어갈 수 있다는 의미
                    if operation_schedule[i].start - operation_schedule[i-1].finish >= op_duration:
                        best_start_time = operation_schedule[i-1].finish
                        return best_start_time + op_duration
                    else:
                        continue
                else:
                    # hole이 있다고 해도 earliest_start로 검증이 필요하다.
                    if operation_schedule[i].start - op_earliest_start >= op_duration:
                        best_start_time = op_earliest_start
                        return best_start_time + op_duration
                    else: 
                        continue
        best_start_time = max(operation_schedule[-1].finish, op_earliest_start)
        return best_start_time + op_duration
            

class JobInfo:
    def __init__(self, name, color, operations, index = None):
        self.name = name
        self.color = color
        self.operation_queue = [Operation(op_info, job_id= str(int(name[4:])-1), color=color, job_index = index) for op_info in operations]
        self.total_duration = sum([op.duration for op in self.operation_queue])

class Job(JobInfo):
    def __init__(self, job_info, index, deadline):
        super().__init__(job_info['name'], job_info['color'], job_info['operations'], index)
        self.index = index
        self.deadline = deadline
        self.estimated_tardiness = 0
        self.tardiness = 0
        self.time_exceeded = 0
        self.is_done = False
        
    def __lt__(self, other):
        # Define the comparison first by estimated_tardiness descending and then by index ascending
        if self.is_done and not other.is_done:
            return False
        if not self.is_done and other.is_done:
            return True
        
        return self.index < other.index
    
    def __str__(self) -> str:
        return f"{self.name} - Repeat {self.index + 1}\t\t:\tTardiness/Deadline = {self.tardiness}/{self.deadline}"
    
class Operation():
    def __init__(self, operation_info, job_id, color, job_index = None):
        self.sequence = None  # 초기화 시점에는 설정되지 않음
        self.index = operation_info['index']
        self.type = type_encoding(operation_info['type'])
        self.duration = operation_info['duration']
        self.predecessor = operation_info['predecessor']
        self.earliest_start = operation_info['earliest_start'] if operation_info['earliest_start'] else 0
        self.start = None  # 초기화 시점에는 설정되지 않음
        self.finish = None  # 초기화 시점에는 설정되지 않음
        self.machine = -1
        self.job = job_id
        self.job_index = job_index
        self.color = color
        self.expected_start = 0
        self.estimated_tardiness = 0

    def to_dict(self):
        return {
            'sequence': self.sequence,
            'index': self.index,
            'type': self.type,
            'predecessor': self.predecessor,
            'earliest_start': self.earliest_start,
            'duration': self.duration,
            'start': self.start,
            'finish': self.finish,
            'machine': self.machine,
            'color': self.color,
            'job': self.job,
            'job_index': self.job_index,
        }

    def __str__(self):
        return f"job : {self.job}, index : {self.index} | ({self.start}, {self.finish})"
    
class customRepeatableSchedulerWithoutETD():
    def __init__(self, jobs, machines, cost_deadline_per_time, cost_hole_per_time, cost_processing_per_time, cost_makespan_per_time, profit_per_time, current_repeats, max_time = 150, num_of_types = 4) -> None:
        self.machines = [Machine(machine_info)
                          for machine_info in machines]
        self.job_infos = [JobInfo(job_info["name"], job_info["color"], job_info["operations"]) for job_info in jobs]
        self.jobs = []
        for job_info in jobs:
            job_list = []
            for i, deadline in enumerate(job_info['deadline']):
                job = Job(job_info, i, deadline)
                job_list.append(job)
            heapq.heapify(job_list)
            self.jobs.append(job_list)

        self.operations = [operation for job_list in self.jobs for job in job_list for operation in job.operation_queue]
        
        len_jobs = len(self.jobs)
        # Reset 할 때 DeepCopy를 위해 원본을 저장해둠
        self.original_jobs = copy.deepcopy(self.jobs)
        self.original_machines = copy.deepcopy(self.machines)
        self.original_operations = copy.deepcopy(
             [operation for job_list in self.jobs for job in job_list for operation in job.operation_queue]
        )
        

        self.cost_deadline_per_time = cost_deadline_per_time
        self.cost_hole_per_time = cost_hole_per_time
        self.cost_processing_per_time = cost_processing_per_time
        self.cost_makespan_per_time = cost_makespan_per_time
        self.profit_per_time = profit_per_time

        self.current_repeats = current_repeats
        
        self.schedule_buffer = [[-1, -1] for _ in range(len_jobs)]
        self.max_time = max_time
        
        # 4 : 각 리소스별 operation 수 최댓값
        # self.original_job_details = np.ones(
        #     (len_jobs, 4, 2), dtype=np.int8) * -1
        # for j in range(len_jobs):
        #     for o in range(len(self.job_infos[j].operation_queue)):
        #         self.original_job_details[j][o][0] = int(
        #             self.job_infos[j].operation_queue[o].duration // 100)
        #         self.original_job_details[j][o][1] = int(
        #             self.job_infos[j].operation_queue[o].type)

        self.job_state = None
        # self.machine_types = None
        self.schedule_heatmap = None
        # self.action_space = spaces.MultiDiscrete([len_machines, len_jobs])
        
        self.action_mask = np.ones(
            shape=(len(self.machines) * len(self.jobs)), dtype=bool)
        self.legal_actions = np.ones(
            shape=(len(self.machines), len(self.jobs)), dtype=bool)

        self.current_schedule = []
        self.num_scheduled_operations = 0
        self.num_steps = 0
        self.last_finish_time = 0
        self.valid_count = 0

        self.machine_operation_rate = np.zeros(
            len(self.machines), dtype=np.float32)

        self.job_term = 0
        self.machine_term = 0

        self.cost_deadline = 0
        self.cost_hole = 0
        self.cost_processing = 0
        self.cost_makespan = 0
        
        # type별 지표 추가
        self.num_of_types = num_of_types
        self.remain_op_duration_per_type = [[] for _ in range(self.num_of_types)]

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # 환경과 관련된 변수들
        self.jobs = copy.deepcopy(self.original_jobs)
        self.machines = copy.deepcopy(self.original_machines)
        self.operations = copy.deepcopy(self.original_operations)
        # self.current_job_details = copy.deepcopy(self.original_job_details)

        self.schedule_buffer = [[-1, -1] for _ in range(len(self.jobs))]


        # 내부 동작을 위한 변수들
        # self.job_state 관한 추가설명 / Job 하나 당 가지는 정보는 아래와 같다
        # 1. 남은 operation 수
        # 2. 다음으로 수행할 Operation의 Type
        # 3. 다음으로 수행할 Operation의 earliest_start
        # 4. 다음으로 수행할 Operation의 duration
        # self.job_state = np.zeros((len(self.jobs), 4), dtype=np.int32)
        # self.machine_types = np.zeros(
        #     (len(self.machines), 25), dtype=np.int8)
        self.schedule_heatmap = np.zeros(
            (len(self.machines), self.max_time), dtype=np.int8)
        # schedule_heatmap의 각 행의 맨 끝 값은 -1로 세팅
        self.schedule_heatmap[:, -1] = -1

        self.legal_actions = np.ones(
            (len(self.machines), len(self.jobs)), dtype=bool)
        self.action_mask = np.ones(
            (len(self.machines) * len(self.jobs)), dtype=bool)

        self.machine_operation_rate = np.zeros(
            len(self.machines), dtype=np.float32)

        # type별 지표 추가
        self.remain_op_duration_per_type = [[] for _ in range(self.num_of_types)]


        self.update_state(None)

        # 기록을 위한 변수들
        self.current_schedule = []
        self.num_scheduled_operations = 0
        self.num_steps = 0
        self.last_finish_time = 0
        self.valid_count = 0

        self.cost_deadline = 0
        self.cost_hole = 0
        self.cost_processing = 0
        self.cost_makespan = 0

        
        return self.get_observation(), self.get_info() 

    def action_masks(self):
        return self.action_mask

    def _update_action_masks(self, action):
        self._update_legal_actions(action)
        result = []  # 빈 리스트
        for legal_action in self.legal_actions:
            result.extend(legal_action)  # 각 legal_action을 result에 추가
        self.action_mask = np.array(result)  # 
        return self.action_mask

    def update_state(self, action=None):
        if action is not None: 
            self.valid_count += 1
            self._update_operation_state(action)
            self._schedule_operation(action)
            self._update_job_state()
            self._update_schedule_buffer()
            self._update_action_masks(action)
            self._update_machine_state(action)
            self.last_finish_time = self._get_final_operation_finish()
        else:
            self._update_operation_state(action)
            self._update_schedule_buffer()
            self._update_machine_state(action)
            self._update_job_state()
            self._update_action_masks(action)

    def _update_operation_state(self, action):
        # action이 없다면 초기화
        if action is None:
            for op in self.operations:
                self.remain_op_duration_per_type[op.type].append(op.duration // 100)
            
            # # test
            # print(self.remain_op_duration_per_type)
        else:
            # self.remain_op_duration_per_type에서 선택된 Job의 operation의 type에서 선택된 operation의 duration을 제거
            selected_job = self.jobs[action[1]][0]
            selected_operation = selected_job.operation_queue[self.schedule_buffer[action[1]][1]]
            self.remain_op_duration_per_type[selected_operation.type].remove(selected_operation.duration // 100)

            # # test
            # print(self.remain_op_duration_per_type)

    def _update_legal_actions(self, action = None):
        # if action:
        #     for machine_index in range(len(self.machines)):
        #         machine = self.machines[machine_index]
        #         job_index = action[1]
        #         if self.schedule_buffer[job_index] == [-1, -1]:
        #             self.legal_actions[:, job_index] = False
        #             break
            
        #         operation_index = self.schedule_buffer[job_index][1]
        #         operation = self.job_infos[job_index].operation_queue[operation_index]
        #         if machine.can_process_operation(operation.type):
        #             self.legal_actions[machine_index, job_index] = True
        #         else:
        #             self.legal_actions[machine_index, job_index] = False
        # else:
        for machine_index in range(len(self.machines)):
            machine = self.machines[machine_index]
            for job_index in range(len(self.jobs)):
                if self.schedule_buffer[job_index] == [-1, -1]:
                    self.legal_actions[:, job_index] = False
                    continue
                operation_index = self.schedule_buffer[job_index][1]
                operation = self.job_infos[job_index].operation_queue[operation_index]
                if machine.can_process_operation(operation.type):
                    self.legal_actions[machine_index, job_index] = True
                else:
                    self.legal_actions[machine_index, job_index] = False

    def _update_machine_state(self, action=None):
        if action is None:
            # for i, machine in enumerate(self.machines):
            #     self.machine_types[i] = [
            #         1 if i in machine.ability else 0 for i in range(25)]
            return
    
        machine = self.machines[action[0]]
        operation_schedule = machine.operation_schedule
        self.schedule_heatmap[action[0]] = np.array(self._schedule_to_array(operation_schedule))

        # 선택된 리소스의 스케줄링된 Operation들
        for machine in self.machines:
            working_time = sum([operation.duration for operation in machine.operation_schedule])
            machine.operation_rate = working_time / self._get_final_operation_finish()
            
        self.machine_operation_rate = np.array([machine.operation_rate for machine in self.machines])

    def _update_job_state(self):
        for job_list in self.jobs:
            for job in job_list:
                remaining_operations = [op for op in job.operation_queue if op.finish is None]

                if not remaining_operations:
                    job.tardiness = job.operation_queue[-1].finish - job.deadline
                    job.time_exceeded = max(0, job.operation_queue[-1].finish - job.deadline)
                    job.estimated_tardiness = float(job.tardiness)
                    job.is_done = True
                    continue
                
                earliest_operation = remaining_operations[0]
                best_finish_times = [
                    machine.cal_best_finish_time(op_earliest_start=earliest_operation.earliest_start, op_type = earliest_operation.type, op_duration = earliest_operation.duration)
                    for machine in self.machines
                ]
                best_finish_times = [time for time in best_finish_times if time != -1]

                # if best_finish_times:
                approx_best_finish_time = int(np.mean(best_finish_times))
                # else:
                #     approx_best_finish_time = 0

                remaining_durations = [op.duration for op in remaining_operations[1:]]
                
                #v0 : Best_finish_time (mean 사용) - Operation deadline
                # operation_deadline = job.deadline - sum(remaining_durations)
                #job.estimated_tardiness = (approx_best_finish_time - operation_deadline) 
                #v1 : Best_finish_time (min 사용) - Operation deadline : V0보다 더 안 좋아서 삭제

                #v2 : Best_finish_time (mean 사용) + Scaled Operation deadline
                # scaled operation deadline 추가
                # (지금까지 걸린 시간 + 자기 duration) / total duration 을 deadline에 곱한다
                scaled_rate = (job.total_duration - sum(remaining_durations)) / job.total_duration

                # scaled_operation_deadline = scaled_rate * job.deadline
                # job.estimated_tardiness = approx_best_finish_time - scaled_operation_deadline
                tardiness = approx_best_finish_time - job.deadline
                job.estimated_tardiness = tardiness * scaled_rate
                
            # Rebuild the heap based on the updated estimated tardiness values
            heapq.heapify(job_list)
    

    def _schedule_to_array(self, operation_schedule):
        # sort 할 필요는 없으나 미래를 위해 보험을 들어놨다고 보면됨.
        operation_schedule.sort(key = lambda x: x.start)
        # def is_in_idle_time(time):
        #     for operation in operation_schedule:
        #         # 머신이 일하고 있는 시간에는 True 반환
        #         if operation.start <= time < operation.finish:
        #             return True
        #     # 머신이 일하고 있는 시간에는 True 반환
        #     return False

        binary_schedule = [0 for _ in range(self.max_time)]
        # op_index_schedule = [0 for _ in range(self.max_time)]
        # op_job_schedule = [0 for _ in range(self.max_time)]
        # job_repeat_schedule = [0 for _ in range(self.max_time)]
        encoded_job_repeat_schedule = [0 for _ in range(self.max_time)]
        # result = [0 for _ in range(max_time + max_time%8)]
        for operation in operation_schedule:
            start = min(self.max_time, operation.start // 100)
            finish = min(self.max_time, operation.finish // 100)

            for i in range(start, finish):
                binary_schedule[i] = 1
                # op_index_schedule[i] = operation.index+1
                # op_job_schedule[i] = int(operation.job)+1
                # job_repeat_schedule[i] = int(operation.job_index)+1
                encoded_job_repeat_schedule[i] = int(operation.job)+1 + int(operation.job_index)*len(self.jobs)
        
        # for i in range(self.max_time):
        #     binary_schedule[i] = is_in_idle_time(i*100)

        # result encoding
        # 8칸씩 끊어서 00010000 -> 16으로 encoding
        # max time을 8로 나눈 것을 올림 할것
        # result_encoded = [sum([result[i*8+j] * 2**j for j in range(8)]) for i in range((max_time+7)//8)]
        # return result_encoded

        # 4차원으로 만들기
        binary_schedule[-1] = -1
        binary_schedule = np.array(binary_schedule)
        # op_index_schedule = np.array(op_index_schedule)
        # op_job_schedule = np.array(op_job_schedule)
        # job_repeat_schedule = np.array(job_repeat_schedule)
        encoded_job_repeat_schedule = np.array(encoded_job_repeat_schedule)

        # 4->1차원으로 축소
        return binary_schedule
        return encoded_job_repeat_schedule

        return [binary_schedule, op_index_schedule, op_job_schedule, job_repeat_schedule]

    def _update_schedule_buffer(self):
        for i in range(len(self.schedule_buffer)):
            # 모든 job instance가 done이면 buffer 갱신
            if all(job.is_done for job in self.jobs[i]):
                self.schedule_buffer[i] = [-1, -1]
                continue
            
            # ETD 대신 deadline을 기준으로 job instance 정렬
            job = heapq.heappop(self.jobs[i])

            # 해당 job instance에서 unscheduled operation 선택
            for j, operation in enumerate(job.operation_queue):
                if operation.finish is None:
                    self.schedule_buffer[i] = [job.index, j]
                    break

            heapq.heappush(self.jobs[i], job)

    def _schedule_operation(self, action):
        # Implement the scheduling logic based on the action
        # You need to update the start and finish times of the operations
        # based on the selected operation index (action) and the current state.

        # Example: updating start and finish times
        selected_machine = self.machines[action[0]]
        selected_job = self.jobs[action[1]][0]
        selected_operation = selected_job.operation_queue[self.schedule_buffer[action[1]][1]]
        #print(selected_operation)
        operation_earliest_start = selected_operation.earliest_start
        
        # Check for predecessor's finish time
        if selected_operation.predecessor is not None:
            predecessor_operation = next(
                op for op in selected_job.operation_queue if op.index == selected_operation.predecessor
            )
            operation_earliest_start = max(operation_earliest_start, predecessor_operation.finish)

        operation_duration = selected_operation.duration
        machine_operations = sorted(
            selected_machine.operation_schedule, key=lambda operation: operation.start
        )

        open_windows = []
        start_window = 0
        last_alloc = 0

        for scheduled_operation in machine_operations:
            machine_init = scheduled_operation.start

            if machine_init > start_window:
                open_windows.append([start_window, machine_init])
            start_window = scheduled_operation.finish

            last_alloc = max(last_alloc, start_window)

        # Fit the operation within the first possible window
        window_found = False
        if operation_earliest_start is None:
            operation_earliest_start = 0

        for window in open_windows:
            # Operation could start before the open window closes
            if operation_earliest_start <= window[1]:
                # Now let's see if it fits there
                potential_start = max(operation_earliest_start, window[0])
                if potential_start + operation_duration <= window[1]:
                    # Operation fits into the window
                    min_earliest_start = potential_start
                    window_found = True
                    break

        # If no window was found, schedule it after the end of the last operation on the machine
        if not window_found:
            if operation_earliest_start > 0:
                min_earliest_start = max(operation_earliest_start, last_alloc)
            else:
                min_earliest_start = last_alloc

        # schedule it
        selected_operation.sequence = self.num_scheduled_operations + 1
        selected_operation.start = min_earliest_start
        selected_operation.finish = min_earliest_start + operation_duration
        selected_operation.machine = action[0]

        self.current_schedule.append(selected_operation)
        selected_machine.operation_schedule.append(selected_operation)
        self.num_scheduled_operations += 1

        # Update the earliest_start for the next operation in the job
        current_op_index = selected_job.operation_queue.index(selected_operation)
        if current_op_index + 1 < len(selected_job.operation_queue):
            next_operation = selected_job.operation_queue[current_op_index + 1]
            next_operation.earliest_start = selected_operation.finish
            return

    def get_observation(self):
        remaining_repeats = []
        cur_estimated_tardiness_per_job = []
        mean_estimated_tardiness_per_job = []
        std_estimated_tardiness_per_job = []
        real_tardiness_per_job = []
        for job_list in self.jobs:
            # 아래 공식 분모 제거
            estimated_tardiness = [job.estimated_tardiness / 100 for job in job_list]
            cur_estimated_tardiness_per_job.append(job_list[0].estimated_tardiness / 100)
            mean_estimated_tardiness_per_job.append(np.mean(estimated_tardiness))
            std_estimated_tardiness_per_job.append(np.std(estimated_tardiness))
            real_tardiness_per_job.append([job.tardiness / 100 for job in job_list])
            remaining_repeats.append(sum([not job.is_done for job in job_list]))

        mean_tardiness_per_job = [np.mean(job_tardiness) for job_tardiness in real_tardiness_per_job]
        std_tardiness_per_job = [np.std(job_tardiness) for job_tardiness in real_tardiness_per_job]
        
        # 스케줄 버퍼에 올라와있는 작업의 정보 계산
        job_deadline = []
        op_duration = []
        schedule_buffer_job_repeat = []
        schedule_buffer_operation_index = []
        earliest_start_per_operation = []
        op_type = []
        num_remaining_op = []
        remaining_working_time = []
        for i, elem in enumerate(self.schedule_buffer):
            schedule_buffer_job_repeat.append(elem[0])
            schedule_buffer_operation_index.append(elem[1])
            if elem[0] == -1:
                earliest_start_per_operation.append(-1)
                job_deadline.append(-1)
                op_duration.append(-1)
                op_type.append(-1)
                num_remaining_op.append(0)
                remaining_working_time.append(0)
            else:
                earliest_start_per_operation.append(self.jobs[i][0].operation_queue[elem[1]].earliest_start // 100)
                job_deadline.append(self.jobs[i][0].deadline // 100)
                op_duration.append(self.jobs[i][0].operation_queue[elem[1]].duration // 100)
                op_type.append(self.jobs[i][0].operation_queue[elem[1]].type)
                num_remaining_op.append(len([op for op in self.jobs[i][0].operation_queue if op.finish is None]))
                # remaining_working_time은 끝나지 않은 op들의 duration의 총합
                remaining_working_time.append(sum([op.duration // 100 for op in self.jobs[i][0].operation_queue if op.finish is None]))

        self.cal_job_deadline_cost()
        self.cal_machine_cost()
        self.cal_entire_cost()

        # 머신 별 hole의 길이 계산
        hole_length_per_machine = [machine.cal_idle_time() // 100 + min(machine.operation_schedule, key = lambda x : x.start).start // 100 if machine.operation_schedule else 0 for machine in self.machines]
        # 머신 별 ablity_encode
        machine_ability = [machine.encode_ability() for machine in self.machines]

        total_count_per_type = [len(self.remain_op_duration_per_type[i]) for i in range(self.num_of_types)]
        # 아래 두 지표에서 empty list 예외 처리 필요
        mean_operation_duration_per_type = [np.mean(self.remain_op_duration_per_type[i]) if self.remain_op_duration_per_type[i] else 0 for i in range(self.num_of_types)]
        std_operation_duration_per_type = [np.std(self.remain_op_duration_per_type[i]) if self.remain_op_duration_per_type[i] else 0 for i in range(self.num_of_types)]

        observation_v4 = {
            # Vaild 행동, Invalid 행동 관련 지표
            'action_masks': self.action_mask,
            # Operation Type별 지표
            "total_count_per_type": np.array(total_count_per_type),
            "mean_operation_duration_per_type": np.array(mean_operation_duration_per_type),
            "std_operation_duration_per_type": np.array(std_operation_duration_per_type),
            # 현 scheduling 상황 관련 지표
            'last_finish_time_per_machine' : np.array([machine.cal_last_finish_time()//100 for machine in self.machines]),
            'machine_ability' : np.array(machine_ability),
            'hole_length_per_machine' : np.array(hole_length_per_machine),
            "machine_utilization_rate": np.array(self.machine_operation_rate),
            'remaining_repeats': np.array(remaining_repeats),
            # schedule_heatmap
            'schedule_heatmap': self.schedule_heatmap,
            # schedule_buffer 관련 지표
            'schedule_buffer_job_repeat': np.array(schedule_buffer_job_repeat),
            'schedule_buffer_operation_index':  np.array(schedule_buffer_operation_index),
            'cur_op_earliest_start': np.array(earliest_start_per_operation),
            'cur_job_deadline': np.array(job_deadline),
            'cur_op_duration': np.array(op_duration),
            'cur_op_type': np.array(op_type),
            "cur_remain_working_time" : np.array(remaining_working_time),
            "cur_remain_num_op" : np.array(num_remaining_op),
            # 추정 tardiness 관련 지표
            'mean_estimated_tardiness_per_job': np.array(mean_estimated_tardiness_per_job),
            'std_estimated_tardiness_per_job' : np.array(std_estimated_tardiness_per_job),
            'cur_estimated_tardiness_per_job' : np.array(cur_estimated_tardiness_per_job),
            # cost 관련 지표
            'current_costs' : np.array([self.cost_deadline, self.cost_hole, self.cost_processing, self.cost_makespan])
        }
        observation_v3 ={
            # Operation Type별 지표
            "total_count_per_type": np.array(total_count_per_type),
            "mean_operation_duration_per_type": np.array(mean_operation_duration_per_type),
            "std_operation_duration_per_type": np.array(std_operation_duration_per_type),
            # 현 scheduling 상황 관련 지표
            'last_finish_time_per_machine' : np.array([machine.cal_last_finish_time()//100 for machine in self.machines]),
            'machine_ability' : np.array(machine_ability),
            'hole_length_per_machine' : np.array(hole_length_per_machine),
            "machine_utilization_rate": np.array(self.machine_operation_rate),
            'remaining_repeats': np.array(remaining_repeats),
            # schedule_buffer 관련 지표
            'schedule_buffer_job_repeat': np.array(schedule_buffer_job_repeat),
            'schedule_buffer_operation_index':  np.array(schedule_buffer_operation_index),
            'cur_op_earliest_start': np.array(earliest_start_per_operation),
            'cur_job_deadline': np.array(job_deadline),
            'cur_op_duration': np.array(op_duration),
            'cur_op_type': np.array(op_type),
            "cur_remain_working_time" : np.array(remaining_working_time),
            "cur_remain_num_op" : np.array(num_remaining_op),
            # 추정 tardiness 관련 지표
            'mean_estimated_tardiness_per_job': np.array(mean_estimated_tardiness_per_job),
            'std_estimated_tardiness_per_job' : np.array(std_estimated_tardiness_per_job),
            'cur_estimated_tardiness_per_job' : np.array(cur_estimated_tardiness_per_job),
            # cost 관련 지표
            'current_costs' : np.array([self.cost_deadline, self.cost_hole, self.cost_processing, self.cost_makespan])
        }
        observation_v2 = {
            # Vaild 행동, Invalid 행동 관련 지표
            'action_masks': self.action_mask,
            # Operation Type별 지표
            "total_count_per_type": np.array(total_count_per_type),
            "mean_operation_duration_per_type": np.array(mean_operation_duration_per_type),
            "std_operation_duration_per_type": np.array(std_operation_duration_per_type),
            # 현 scheduling 상황 관련 지표
            'last_finish_time_per_machine' : np.array([machine.cal_last_finish_time()//100 for machine in self.machines]),
            'machine_ability' : np.array(machine_ability),
            'hole_length_per_machine' : np.array(hole_length_per_machine),
            "machine_utilization_rate": np.array(self.machine_operation_rate),
            'remaining_repeats': np.array(remaining_repeats),
            # schedule_buffer 관련 지표
            'schedule_buffer_job_repeat': np.array(schedule_buffer_job_repeat),
            'schedule_buffer_operation_index':  np.array(schedule_buffer_operation_index),
            'cur_op_earliest_start': np.array(earliest_start_per_operation),
            'cur_job_deadline': np.array(job_deadline),
            'cur_op_duration': np.array(op_duration),
            'cur_op_type': np.array(op_type),
            "cur_remain_working_time" : np.array(remaining_working_time),
            "cur_remain_num_op" : np.array(num_remaining_op),
            # 추정 tardiness 관련 지표
            'mean_estimated_tardiness_per_job': np.array(mean_estimated_tardiness_per_job),
            'std_estimated_tardiness_per_job' : np.array(std_estimated_tardiness_per_job),
            'cur_estimated_tardiness_per_job' : np.array(cur_estimated_tardiness_per_job),
            # cost 관련 지표
            'current_costs' : np.array([self.cost_deadline, self.cost_hole, self.cost_processing, self.cost_makespan])
        }
        observation_v1 = {
            # Vaild 행동, Invalid 행동 관련 지표
            'action_masks': self.action_mask,
            # Operation Type별 지표
            "total_count_per_type": np.array(total_count_per_type),
            "mean_operation_duration_per_type": np.array(mean_operation_duration_per_type),
            "std_operation_duration_per_type": np.array(std_operation_duration_per_type),
            # 현 scheduling 상황 관련 지표
            'last_finish_time_per_machine' : np.array([machine.cal_last_finish_time()//100 for machine in self.machines]),
            'machine_ability' : np.array(machine_ability),
            'hole_length_per_machine' : np.array(hole_length_per_machine),
            'schedule_heatmap': self.schedule_heatmap,
            # "machine_utilization_rate": np.array(self.machine_operation_rate),
            'mean_real_tardiness_per_job': np.array(mean_tardiness_per_job),
            'std_real_tardiness_per_job': np.array(std_tardiness_per_job),
            'remaining_repeats': np.array(remaining_repeats),
            # schedule_buffer 관련 지표
            'schedule_buffer_job_repeat': np.array(schedule_buffer_job_repeat),
            'schedule_buffer_operation_index':  np.array(schedule_buffer_operation_index),
            'earliest_start_per_operation': np.array(earliest_start_per_operation),
            'job_deadline': np.array(job_deadline),
            'op_duration': np.array(op_duration),
            'op_type': np.array(op_type),
            # 추정 tardiness 관련 지표
            'mean_estimated_tardiness_per_job': np.array(mean_estimated_tardiness_per_job),
            'std_estimated_tardiness_per_job' : np.array(std_estimated_tardiness_per_job),
            'cur_estimated_tardiness_per_job' : np.array(cur_estimated_tardiness_per_job),
            # cost 관련 지표
            'cost_factor_per_time': np.array([self.cost_deadline_per_time, self.cost_hole_per_time, self.cost_processing_per_time, self.cost_makespan_per_time]),
            'current_costs' : np.array([self.cost_deadline, self.cost_hole, self.cost_processing, self.cost_makespan])
        }
        
        return observation_v4

    def test_cal_best_finish_time(self):
        # machine 0에 대해서만 테스트
        machine = self.machines[0]
        for i, elem in enumerate(self.schedule_buffer):
            operation = self.jobs[i][0].operation_queue[elem[1]]
            print(f"Job {i} - Operation {elem[1]} Best Finish Time : {machine.cal_best_finish_time(operation.duration, operation.type, operation.earliest_start)}")

    def get_info(self):
        return {
            'jobs' : self.jobs,
            'finish_time': self.last_finish_time,
            'legal_actions': self.legal_actions,
            'action_mask': self.action_mask,
            'machine_score': self.machine_term,
            'machine_operation_rate': [machine.operation_rate for machine in self.machines],
            'schedule_buffer': self.schedule_buffer,
            'job_estimated_tardiness': [job.estimated_tardiness for job_list in self.jobs for job in job_list],
            'current_schedule': self.current_schedule,
            'job_deadline': [job.deadline for job_list in self.jobs for job in job_list],
            'job_time_exceeded': [job.time_exceeded for job_list in self.jobs for job in job_list],
            'job_tardiness': [job.tardiness for job_list in self.jobs for job in job_list],
            'cost_deadline': self.cost_deadline,
            'cost_hole': self.cost_hole,
            'cost_processing': self.cost_processing,
            'cost_makespan': self.cost_makespan,
            'heatmap': self.schedule_heatmap,
        }
    
    def render(self, mode="seaborn", num_steps=0):
        current_schedule = [operation.to_dict() for operation in self.current_schedule]
        scheduled_df = list(filter(lambda operation: operation['sequence'] is not None, current_schedule))
        scheduled_df = pd.DataFrame(scheduled_df)

        fig, ax = plt.subplots(figsize=(12, 6))

        if scheduled_df.empty:
            ax.set_title("Operation Schedule Visualization")
        else:
            n_machines = len(self.machines)
            ax.set_title(f'Operation Schedule Visualization | steps = {num_steps}')
            ax.set_yticks(range(n_machines))
            ax.set_yticklabels([f'Machine {i}\n ability:{self.machines[i].ability}' for i in range(n_machines)])

            ax.set_xlim(0, self.last_finish_time)
            ax.set_ylim(-1, n_machines)

            legend_jobs = []

            for i in range(len(self.machines)):
                machine_operations = scheduled_df[scheduled_df['machine'] == i]
                for index, operation in machine_operations.iterrows():
                    base_color = mcolors.to_rgba(operation['color'])
                    job_index = operation["job_index"]
                    shade_factor = (job_index + 1) / (len(self.jobs[0]) + 1)
                    shaded_color = tuple([min(max(shade * shade_factor, 0), 1) if idx < 3 else shade for idx, shade in enumerate(base_color)])

                    # Build operation sequence string
                    operation_sequences = scheduled_df[(scheduled_df['job'] == operation['job']) & (scheduled_df['job_index'] == operation['job_index'])].sort_values(by='sequence')
                    operation_info = ' -> '.join(map(str, operation_sequences['machine'].tolist()))

                    job_label = f'Job {int(operation["job"]) + 1} - Repeat{job_index + 1} - ({operation_info})'
                    legend = (int(operation["job"]) + 1, job_label, shaded_color)
                    if legend not in legend_jobs:
                        legend_jobs.append(legend)

                    op_block = mpatches.Rectangle(
                        (operation["start"], i - 0.5), operation["finish"] - operation["start"], 1, facecolor=shaded_color, edgecolor='black', linewidth=1)
                    ax.add_patch(op_block)

            # Add legend for job repetition
            legend_jobs.sort()
            legend_patches = []
            for _, label, color in legend_jobs:
                legend_patches.append(mpatches.Patch(color=color, label=label))
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')

        if mode == 'rgb_array':
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            return np.array(img)
        else:
            plt.show()
    def is_legal(self, action):
        return self.action_mask[action[0]*len(self.jobs) + action[1]]
        return self.legal_actions[action[0], action[1]]

    def is_done(self):
        # 모든 Job의 Operation가 종료된 경우 Terminated를 True로 설정한다
        # 또한 legal_actions가 전부 False인 경우도 Terminated를 True로 설정한다
        return all([job.is_done for job_list in self.jobs for job in job_list])
        return not np.any(self.action_mask) or all([job.operation_queue[-1].finish is not None for job_list in self.jobs for job in job_list])

    def _get_final_operation_finish(self):
        if self.current_schedule:
            return max(self.current_schedule, key=lambda x: x.finish).finish
        else:
            return 0

    def calculate_step_reward(self, action):
        # 머신 가동률의 평균을 reward로 사용
        return np.mean(self.machine_operation_rate)

        # 아래 reward 계산은 의미가 없다고 판단
        # selected_machine = self.machines[action[0]]
        # selected_job = self.jobs[action[1]][0]

        # reward = 0

        # self.update_state(action)
        # # 선택된 job이 방금 끝난 경우
        # if selected_job.is_done:
        #     profit = (selected_job.total_duration) * 10
        #     cost = selected_job.time_exceeded * 5
        #     reward += (profit - cost) / profit

        # # 선택된 machine이 hole이 있는 경우
        # idle_time = selected_machine.cal_idle_time()
        # reward -= idle_time // 100
  
        # return reward
    
        # self.machine_term = 0.0
        # if np.any(self.machine_operation_rate):
        #     self.machine_term = np.mean(self.machine_operation_rate)
        # return self.machine_term

    def calculate_final_reward(self):
        total_up_time = 0
        for i, repeat in enumerate(self.current_repeats):
            total_duration_per_job = self.jobs[i][0].total_duration // 100
            total_up_time += total_duration_per_job * repeat

        profit = total_up_time * self.profit_per_time
        cost = self.cal_final_cost()
        return ((profit - cost) / profit) * 100

    def cal_job_deadline_cost(self):
        sum_of_time_exceed = 0
        deadline_cost_per_job = []
        for job_list in self.jobs:
            deadline_cost = [job.time_exceeded for job in job_list]
            deadline_cost_per_job.append(sum(deadline_cost))
            sum_of_time_exceed += sum(deadline_cost)
        self.cost_deadline = sum_of_time_exceed / 100 * self.cost_deadline_per_time

        return self.cost_deadline
    
    def cal_machine_cost(self):
        up_time_cost_per_machine = []
        hole_time_cost_per_machine = []
        sum_of_hole_time = 0
        sum_of_up_time = 0
        for machine in self.machines:
            up_time = 0
            hole_time = 0
            if not machine.operation_schedule:
                continue
            machine.operation_schedule.sort(key = lambda x: x.start)
            first_start = machine.operation_schedule[0].start
            last_finish = machine.operation_schedule[-1].finish

            for op in machine.operation_schedule:
                up_time += op.duration

            hole_time = last_finish - first_start - up_time
            sum_of_up_time += up_time
            sum_of_hole_time += hole_time

            up_time_cost_per_machine.append(up_time)
            hole_time_cost_per_machine.append(hole_time)
        
        self.cost_hole = sum_of_hole_time * self.cost_hole_per_time / 100
        self.cost_processing = sum_of_up_time * self.cost_processing_per_time / 100

        return self.cost_hole + self.cost_processing

    def cal_entire_cost(self):
            self.cost_makespan = self._get_final_operation_finish() * self.cost_makespan_per_time / 100
            return self.cost_makespan

    def cal_final_cost(self):
        # 반도체 공장
        # 주문: 만들어야하는 chip의 종류가 1~12이고 하루에 처리해야하는 일의 양이 변동함. (정규분포 따름)
        # 기계: 1~8번까지 있고, 각 기계가 할 수 있는 일의 양이 다름
        # Cost 

        # 1. Job deadline 어긴 정도 / 비율로 duration이 분모로감
        # - 총 duration 600, deadline : 900
        # - 끝난 시간 1000이면 1/6 * 단위시간 당 cost 만큼 cost 발생
        # -> obs 추가
        job_deadline_cost = self.cal_job_deadline_cost()
        # 2. machine Processing cost, hole cost 는 절반(hyperparams)으로
        # - 100 times 일할때마다 0.1,  
        # - 100 times 놀 때마다 0.05, 
        # 시작되는 시점부터 머신은 가동됨
        machine_util_cost = self.cal_machine_cost()
        # 3. Entire cost, (makespan cost)
        # - Makespan 100단위당 0.01
        entire_cost = self.cal_entire_cost()

        return job_deadline_cost + machine_util_cost + entire_cost