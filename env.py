import torch
from dataclasses import dataclass

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_random_problems(batch_size, stage_cnt, machine_cnt, job_cnt, process_time_params):
    time_low = process_time_params['time_low']
    time_high = process_time_params['time_high']

    problems_INT_list = []
    for stage_num in range(stage_cnt):
        stage_problems_INT = torch.randint(low=time_low, high=time_high, size=(batch_size, job_cnt, machine_cnt))
        problems_INT_list.append(stage_problems_INT)

    return problems_INT_list

@dataclass
class Step_State:
    time: int = 0
    stage : int = 0
    able: torch.Tensor = None
    # shape: (B, )
    mask: torch.Tensor = None
    # shape: (B, I, J)

class FFSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.B = env_params['batch_size']
        self.S = env_params['stage_cnt']
        self.J = env_params['machine_cnt']
        self.total_machine_cnt = self.J * self.S
        self.I = env_params['job_cnt']
        self.process_time_params = env_params['process_time_params']
        # self.rollout_size = env_params['rollout']
        self.done = None

    def reset(self):
        # problem generation
        problems_INT_list = get_random_problems(1, self.S, self.J, self.I, self.process_time_params)
        # list : Stage * [B, I, J]

        self.job_durations = torch.empty(size=(self.B, self.S, self.I, self.J),
                                            dtype=torch.long)
        problems_list = []
        for stage_num in range(self.S):
            stage_problems_INT = problems_INT_list[stage_num]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            stage_problems = stage_problems.expand(self.B, self.I, self.J)
            problems_list.append(stage_problems)
            self.job_durations[:, stage_num, :, :] = stage_problems

        init_state = Step_State()

        self.schedule = torch.full(size=(self.B, self.S, self.J, self.I),
                                   dtype=torch.long, fill_value=-999999)
        self.machine_wait_step = torch.zeros(size=(self.B, self.S, self.J),
                                        dtype=torch.long)
        # shape: (B, S, J)
        self.job_location = torch.zeros(size=(self.B, self.I), dtype=torch.long)
        # shape: (B, I)
        self.job_wait_step = torch.zeros(size=(self.B, self.S, self.I), dtype=torch.long)
        # shape: (B, I)
        self.job_assign = torch.full(size=(self.B, self.S, self.I),
                            fill_value=float('-inf'))
        self.job_assign[...] = 0
        # shape: (B, S, I)
        mask = torch.full(size=(self.B, self.I, self.J),
                    fill_value=float('-inf'))
        mask[...] = 0
        init_state.mask = mask
        init_state.able = torch.ones(size=(self.B,),dtype=torch.bool)
        # shape: (B, )

        self.done = None
        self.reward = None
        return init_state, self.reward, self.done, problems_list

    def step(self, state, action):
        action_I = action // self.J
        action_J = action % self.J

        stage_num = state.stage
        able = state.able
        time = state.time
        setect_duration = self.job_durations[able, stage_num, action_I, action_J]
        self.job_assign[able, stage_num, action_I] = float('-inf')

        self.schedule[able, stage_num, action_J, action_I] = time
        self.job_wait_step[able, stage_num, action_I] += setect_duration
        self.machine_wait_step[able, stage_num, action_J] = time + setect_duration

        # job able check
        not_assign_job = self.job_assign[:, stage_num, :] == 0
        current_stage = self.job_location == stage_num
        able_job = current_stage & not_assign_job

        # machine able
        able_machine = (self.machine_wait_step[:, stage_num, :] <= state.time)

        # able machine and task matrix
        able_mask = able_job[:, :, None].expand(self.B, self.I, self.J) \
                                & able_machine[:, None, :].expand(self.B, self.I, self.J)
        # shape : (B, I, J)
        able_batch = able_machine.any(dim=1) & able_job.any(dim=1)

        while (~able_batch).all():
            # stage update
            stage_num += 1

            if stage_num == self.S:
                # time step update
                time += 1
                self.job_wait_step[self.job_assign != 0] -= 1
                self.job_location[((self.job_assign != 0) & (self.job_wait_step == 0)).any(dim=1)] += 1
                # stage reset
                stage_num = 0

            ## next stage able check
            # job able
            not_assign_job = self.job_assign[:, stage_num, :] == 0
            current_stage = self.job_location == stage_num
            able_job = current_stage & not_assign_job

            # machine able
            able_machine = (self.machine_wait_step[:, stage_num, :] <= time)

            # able machine and task matrix
            able_mask = able_job[:, :, None].expand(self.B, self.I, self.J) \
                                & able_machine[:, None, :].expand(self.B, self.I, self.J)
            # shape : (B, I, J)

            able_batch = able_machine.any(dim=1) & able_job.any(dim=1)

            if (self.job_assign != 0).all():
                self.done = True
                self.reward = self._get_makespan()[0] # yet
                break

        mask = torch.full(size=(self.B, self.I, self.J),
                            fill_value=float('-inf'))
        mask[able_mask] = 0
        next_state = Step_State()
        next_state.time = time
        next_state.stage = stage_num
        next_state.mask = mask
        next_state.able = able_batch
        return next_state, self.reward, self.done

    def _get_makespan(self):
        job_durations = self.job_durations.permute(0, 1, 3, 2)
        job_durations = job_durations.reshape(self.B, self.S * self.J, self.I)

        schedule = self.schedule.reshape(self.B, self.S * self.J, self.I)
        end_schedule = schedule + job_durations
        end_time_max, _ = end_schedule.max(dim=2)
        end_time_max = end_time_max.max(dim=1)
        return end_time_max

    def draw_gantt_chart(self, b):
        job_durations = self.job_durations[b, ...]
        schedule = self.schedule[b, :, :, :]
        makespan = self._get_makespan()[0][b]
        fig,ax = plt.subplots(figsize=(makespan/3, 5))
        cmap = self._get_cmap()
        plt.xlim(0, makespan)
        plt.ylim(0, self.J * self.S)
        ax.invert_yaxis()

        plt.plot([0, makespan], [self.J, self.J], 'black')
        plt.plot([0, makespan], [self.J * self.S, self.J * self.S], 'black')

        for stage in range(self.S):
            for machine in range(self.J):
                # print(stage, machine)
                duration = job_durations[stage, :, machine]
                machine_schedule = schedule[stage, machine, :]
                for i in range(self.I):
                    job_length = duration[i].item()
                    job_start_time = machine_schedule[i].item()
                    if job_start_time >= 0:
                        rect = patches.Rectangle((job_start_time,(machine + self.J * stage)),job_length,1, facecolor=cmap(i))
                        ax.add_patch(rect)

                        rx, ry = rect.get_xy()
                        cx = rx + rect.get_width()/2.0
                        cy = ry + rect.get_height()/2.0
                        ax.annotate(f"{i}", (cx, cy), color='black',  fontsize=15, ha='center', va='center')
        ax.grid()
        ax.set_axisbelow(True)
        plt.show()

    def _get_cmap(self):
        colors_list = ['red', 'orange', 'yellow', 'green', 'blue',
                        'purple', 'aqua', 'aquamarine', 'black',
                        'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate',
                        'coral', 'cornflowerblue', 'darkblue', 'darkgoldenrod', 'darkgreen',
                        'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                        'darkorchid', 'darkred', 'darkslateblue', 'darkslategrey', 'darkturquoise',
                        'darkviolet', 'deeppink', 'deepskyblue', 'dimgrey', 'dodgerblue',
                        'forestgreen', 'gold', 'goldenrod', 'gray', 'greenyellow',
                        'hotpink', 'indianred', 'khaki', 'lawngreen', 'magenta',
                        'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                        'mediumpurple',
                        'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                        'navy', 'olive', 'olivedrab', 'orangered',
                        'orchid',
                        'palegreen', 'paleturquoise', 'palevioletred', 'pink', 'plum', 'powderblue',
                        'rebeccapurple',
                        'rosybrown', 'royalblue', 'saddlebrown', 'sandybrown', 'sienna',
                        'silver', 'skyblue', 'slateblue',
                        'springgreen',
                        'steelblue', 'tan', 'teal', 'thistle',
                        'tomato', 'turquoise', 'violet', 'yellowgreen']

        cmap = ListedColormap(colors_list, N=self.I)
        return cmap