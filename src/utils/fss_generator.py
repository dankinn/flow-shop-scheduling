# Copyright 2024 D-Wave Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append("./src")

import numpy as np

from dwave.optimization.mathematical import maximum
from dwave.optimization.model import Model
from dwave.system import LeapHybridNLSampler

from model_data import JobShopData
from utils.utils import _2d_nonnegative_int_array


class Job:
    """Job with a list of processing times for each machine, in the
    form {machine: processing_time}.
    """
    def __init__(self, id_: int, processing_times: dict[int]):
        self.id_ = id_
        self.processing_times = processing_times

    def __repr__(self):
        return f"Job {self.id_}"

    def __str__(self):
        return f"Job {self.id_}"
    
    def __eq__(self, other):
        if not isinstance(other, Job):
            return False
        return self.id_ == other.id_
    
    def __hash__(self) -> int:
        return hash('job_' + str(self.id_))
    

class Machine:
    """A machine object to execute jobs."""
    def __init__(self, id_: str):
        self.id = id_

    def __repr__(self):
        return f"Machine: {self.id}"

    def __str__(self):
        return f"Machine: {self.id}"
    
    def __eq__(self, other):
        if not isinstance(other, Machine):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash('machine_' + str(self.id))
    

class CooldownPeriod:
    """A cooldown period associated with a specific Machine."""
    def __init__(self, machine: Machine, durations: dict[Job, int]):
        """Initialize a cooldown period for a machine.

        Args:
            machine (Machine): The machine to which the cooldown period
                applies.
            durations (dict[Job, int]): A dictionary of jobs and their
                associated cooldown durations, in the form {job: duration}.
        """
        self.machine = machine
        self.durations = durations

    def __repr__(self):
        return f"Cooldown period: {self.durations}"

    def __str__(self):
        return f"Cooldown period: {self.durations}"
    
    
class ScheduledJob:
    """A scheduled job with start and end times."""
    def __init__(self, job: Job, start_time: int, end_time: int, cooldown_time: int=None):
        self.job = job
        self.start_time = start_time
        self.end_time = end_time
        self.cooldown_time = cooldown_time

    @property
    def duration(self):
        return self.end_time - self.start_time

    def __repr__(self):
        if self.cooldown_time is not None:
            return f"Scheduled job: {self.job.id_} ({self.start_time}, {self.end_time}); cooldown: {self.cooldown_time}"
        return f"Scheduled job: {self.job.id_} ({self.start_time}, {self.end_time})"

    def __str__(self):
        if self.cooldown_time is not None:
            return f"Scheduled job: {self.job.id_} ({self.start_time}, {self.end_time}); cooldown: {self.cooldown_time}"
        return f"Scheduled job: {self.job.id_} ({self.start_time}, {self.end_time})"
    
    def __eq__(self, other):
        if not isinstance(other, ScheduledJob):
            return False
        return self.job == other.job \
                and self.start_time == other.start_time \
                and self.end_time == other.end_time \
                and self.cooldown_time == other.cooldown_time
    
    def __hash__(self) -> int:
        return hash('scheduled_job_' + str(self.job.id_))


class FlowShopScheduler:
    '''
    This class is used to generate a model encoding a flow-shop scheduling problem.
    '''

    def __init__(self,
                 machines: list[Machine],
                 jobs: list[Job], 
                 cooldown_periods: list[CooldownPeriod]=[]):
        """Initialize a flow-shop scheduler.

        Args:
            machines (list[Machine]): A list of machines to execute jobs. The order of this list
                indicates the order all jobs must follow.
            jobs (list[Job]): A list of jobs to be executed on the machines. The order of this list
                is inconsequential.
            cooldown_periods (list[CooldownPeriod], optional): A list of
                cooldown periods associated with specific machines. Defaults to []. The order of
                this list is inconsequential.
        """
        if len(machines) == 0:
            raise ValueError("At least one machine is required")
        if len(jobs) == 0:
            raise ValueError("At least one job is required")
        self.machines = machines
        self.jobs = jobs
        self.cooldown_periods = cooldown_periods

        self.ordering_groups = []
        last_idx = 0
        self.cooldown_periods.sort(key=lambda x: self.machines.index(x.machine))
        for cooldown_period in self.cooldown_periods:
            if cooldown_period.machine not in self.machines:
                raise ValueError("Cooldown period machine not in machines list")
            machine_idx = self.machines.index(cooldown_period.machine)
            self.ordering_groups.append(machines[last_idx:machine_idx+1])
            last_idx = machine_idx + 1

        self.ordering_groups.append(machines[last_idx:])


    def generate_integer_bounds(self):
        """Generate integer bounds for the flow-shop scheduling problem.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of 2D arrays representing the integer
                bounds for the flow-shop scheduling problem.
        """
        ordering_group_bounds = [[None, None] for _ in range(len(self.ordering_groups)-1)]
        min_time = 0
        total_time = 0
        for idx, ordering_group in enumerate(self.ordering_groups[:-1]):
            ordering_group_bounds[idx][0] = min_time
            min_time_nominee = float('inf')
            for job in self.jobs:
                job_time = 0
                for machine in ordering_group:
                    job_time += job.processing_times[machine]
                job_time += self.cooldown_periods[idx].durations[job]
                total_time += job_time
                if job_time < min_time_nominee:
                    min_time_nominee = job_time
            min_time += min_time_nominee
            ordering_group_bounds[idx][1] = total_time
        return ordering_group_bounds


    def build_model(self):
        """Build the flow-shop scheduling model.

        Returns:
            Tuple[Model, List[ArrayObserver]]: A model encoding the flow-shop
            scheduling problem and the list of end-times for the problem solution.
        """
        self.model = Model()
        
        processing_times = [[job.processing_times[machine] for job in self.jobs] \
                            for machine in self.machines]
        processing_times = next(_2d_nonnegative_int_array(processing_times=processing_times))
        times = self.model.constant(processing_times)

        cooldown_times = [[x.durations[job] for job in self.jobs] for x in self.cooldown_periods]
        cooldown_times = self.model.constant(cooldown_times)

        num_jobs = len(self.jobs)
        self.orders = [self.model.list(num_jobs) for _ in range(len(self.ordering_groups))]
        ordering_group_bounds = self.generate_integer_bounds()
        self.order_group_ends = [self.model.integer(shape=num_jobs, lower_bound=lb, upper_bound=ub) \
                                 for [lb, ub] in ordering_group_bounds]

        self.end_times = []
        for order_idx, machines in enumerate(self.ordering_groups):
            order = self.orders[order_idx]
            order_end_times = []
            for machine_idx in range(len(machines)):
                machine_m = self.machines.index(machines[machine_idx])
                machine_m_times = []
                if order_idx == 0:
                    for job_j in range(len(self.jobs)): #we'll iterate through the order of the first grouping
                        if machine_idx == 0:
                            if job_j == 0: #if a job is the first in order, then it'll start at time 0 and end at the processing time
                                machine_m_times.append(times[machine_m, :][order[job_j]])
                            else: # otherwise, the job will start at the prior job's end time
                                end_job_j = times[machine_m][order[job_j]]
                                end_job_j += machine_m_times[-1]
                                machine_m_times.append(end_job_j)
                        else:
                            if job_j == 0:
                                end_job_j = order_end_times[machine_m - 1][job_j]
                                end_job_j += times[machine_m, :][order[job_j]]
                                machine_m_times.append(end_job_j)
                            else:
                                end_job_j = maximum(order_end_times[machine_m - 1][job_j], machine_m_times[-1])
                                end_job_j += times[machine_m, :][order[job_j]]
                                machine_m_times.append(end_job_j)
                        
                
                        if (machine_idx == len(machines) - 1) and (len(self.ordering_groups) > 1): 
                            #if there are cooldown periods, we'll add the cooldown time
                            self.model.add_constraint(self.order_group_ends[order_idx][order[job_j]] == \
                                                    machine_m_times[-1] + cooldown_times[order_idx][order[job_j]])
                    
                else:
                    for job_j in range(len(self.jobs)): #iterate through this groups ordering
                        if machine_idx == 0:
                            if job_j == 0:
                                end_job_j = self.order_group_ends[order_idx-1][order[job_j]]
                                end_job_j += times[machine_m][order[job_j]]
                                machine_m_times.append(end_job_j)
                            else:
                                end_job_j = maximum(self.order_group_ends[order_idx-1][order[job_j]],
                                                        machine_m_times[-1],
                                                        )
                                end_job_j += times[machine_m][order[job_j]]
                                machine_m_times.append(end_job_j)
                        else:
                            if job_j == 0:
                                end_job_j = order_end_times[-1][job_j]
                                end_job_j += times[machine_m][order[job_j]]
                                machine_m_times.append(end_job_j)
                            else:
                                end_job_j = maximum(self.order_group_ends[order_idx-1][order[job_j]],
                                                        machine_m_times[-1],
                                                        )
                                end_job_j += times[machine_m][order[job_j]]
                                machine_m_times.append(end_job_j)

                        if (machine_idx == len(machines) - 1) and (len(self.ordering_groups) > order_idx + 1):
                            #if this is the last machine in this ordering group, we'll add the cooldown time
                            self.model.add_constraint(self.order_group_ends[order_idx][order[job_j]] == \
                                                    machine_m_times[-1] + cooldown_times[order_idx][order[job_j]])
                order_end_times.append(machine_m_times)
            self.end_times.append(order_end_times)

        # The objective is to minimize the last end time
        makespan = maximum(self.end_times[-1][-1])
        self.model.minimize(makespan)
        self.model.lock()


    def _calculate_end_times(self) -> None:
        """Calculate the end-times for the FSS job results.

        Helper function to calculate the end-times for the FSS job results obtained 
        from the NL Solver. The result objects will be saved as dictionaries of
        ScheduledJob objects. 

        Update when symbol labels are supported.

        Modifies:
            self.job_schedules: A dictionary of jobs and their scheduled times.
            self.machine_schedules: A dictionary of machines and their scheduled times.
        """
        group_orders = [order.state() for order in self.orders]

        self.job_schedules = {job: [] for job in self.jobs}
        self.machine_schedules = {machine: [] for machine in self.machines}
        for order_idx, group_order in enumerate(group_orders):
            for group_idx, job_idx in enumerate(group_order):
                job = self.jobs[int(job_idx)]
                for machine_idx, machine in enumerate(self.ordering_groups[order_idx]):
                    if order_idx == 0 and group_idx == 0 and machine_idx == 0:
                        start_time = 0
                    elif group_idx == 0:
                        job_last_machine_end = self.job_schedules[job][-1].end_time + \
                            self.job_schedules[job][-1].cooldown_time
                        start_time = job_last_machine_end
                    else:
                        if len(self.job_schedules[job]) == 0:
                            job_last_machine_end = 0
                        else:
                            job_last_machine_end = self.job_schedules[job][-1].end_time + \
                                self.job_schedules[job][-1].cooldown_time
                        machine_last_job_end = self.machine_schedules[machine][-1].end_time
                        start_time = np.maximum(job_last_machine_end, machine_last_job_end)

                    end_time = start_time + job.processing_times[machine]
                    if (machine_idx == len(self.ordering_groups[order_idx]) - 1) and \
                        (order_idx < len(self.ordering_groups) - 1):
                        cooldown_time = self.cooldown_periods[order_idx].durations[job]
                    else:
                        cooldown_time = 0

                    scheduled_job = ScheduledJob(job, start_time, end_time, cooldown_time)
                    self.job_schedules[job].append(scheduled_job)
                    self.machine_schedules[machine].append(scheduled_job)


    def solve(self, time_limit: int=None, sampler_kwargs: dict={}) -> None:
        """Solve the flow-shop scheduling problem.

        Args:
            time_limit (int, optional): Time limit in seconds. Defaults to None.
            **sampler_kwargs: Additional keyword arguments for the sampler.

        """
        sampler = LeapHybridNLSampler(**sampler_kwargs)
        sampler.sample(self.model, time_limit=time_limit, label="FSS demo")

        self._calculate_end_times()


    def __repr__(self):
        return f"FlowShopScheduler with {len(self.jobs)} jobs and {len(self.machines)} machines"

    def __str__(self):
        return f"FlowShopScheduler with {len(self.jobs)} jobs and {len(self.machines)} machines"
    

def create_fss_from_processing_times(processing_times: np.ndarray, 
                                     cooldown_times: dict[int, np.ndarray]) \
                                        -> FlowShopScheduler:
    """Create a FlowShopScheduler object from processing times and cooldown 
        times.

    Args:
        processing_times (np.ndarray): A 2D array of processing times 
            for each job on each machine.
        cooldown_times (dict[int, np.ndarray]): A dictionary of cooldown 
            times for each machine.
    """
    machines = [Machine(i) for i in range(processing_times.shape[0])]
    jobs = [Job(i, {machines[j]: processing_times[j, i] for j in range(processing_times.shape[0])}) \
             for i in range(processing_times.shape[1])]
    cooldown_periods = [CooldownPeriod(machines[i], {jobs[j]: cooldown_times[i][j] \
                                                     for j in range(cooldown_times[i].shape[0])}) \
                                                     for i in cooldown_times.keys()]
    fss = FlowShopScheduler(machines, jobs, cooldown_periods)
    return fss


def create_random_fss(num_machines: int,
                      num_jobs: int,
                      num_cooldown_periods: int,
                      max_processing_time: int,
                      max_cooldown_time: int,
                      min_processing_time: int=1,
                      min_cooldown_time: int=1,
                      seed: int=None) -> FlowShopScheduler:
    """Create a random FlowShopScheduler object. 

    Args:
        num_machines (int): The number of machines.
        num_jobs (int): The number of jobs.
        num_cooldown_periods (int): The number of cooldown periods.
        max_processing_time (int): The maximum processing time for a job.
        max_cooldown_time (int): The maximum cooldown time for a machine.
        min_processing_time (int, optional): The minimum processing time for a job.
            Defaults to 1.
        min_cooldown_time (int, optional): The minimum cooldown time for a machine. 
            Defaults to 1.
        seed (int, optional): Random seed, which can be used for generating reproducible 
            results. Defaults to None.

    Returns:
        FlowShopScheduler: A FlowShopScheduler object with random parameters, bounded by the 
            given arguments.
    """
    if min_processing_time > max_processing_time:
        raise ValueError("Minimum processing time must be less than maximum processing time")
    if min_cooldown_time > max_cooldown_time:
        raise ValueError("Minimum cooldown time must be less than maximum cooldown time")
    if num_cooldown_periods > num_machines:
        raise ValueError("""Number of cooldown periods must be less than or equal to the 
                         number of machines""")

    np.random.seed(seed)
    processing_times = np.random.randint(min_processing_time, max_processing_time, 
                                         size=(num_machines, num_jobs))
    cooldown_periods = np.random.choice(num_machines, size=num_cooldown_periods, replace=False)
    cooldown_times = {i: np.random.randint(min_cooldown_time, max_cooldown_time, size=num_jobs) \
                      for i in cooldown_periods}

    fss = create_fss_from_processing_times(processing_times, cooldown_times)
    return fss


def create_fss_from_file(file_path: str,
                         cooldown_times={}) -> FlowShopScheduler:
    """Create a FlowShopScheduler object from a file.

    Args:
        file_path (str): The path to the file containing the FSS data.
        cooldown_times (dict, optional): A dictionary of cooldown times for each machine. 
            Defaults to {}.

    Returns:
        FlowShopScheduler: A FlowShopScheduler object with data from the file.
    """
    job_data = JobShopData()
    job_data.load_from_file(file_path)
    fss = create_fss_from_processing_times(processing_times=job_data.processing_times,
                                           cooldown_times=cooldown_times)
    return fss
                      

if __name__ == '__main__':

    # from job_shop_scheduler import JobShopSchedulingModel
    # random_fss = create_random_fss(4, 3, 2, 10, 10, seed=0)
    # random_fss.build_model()
    # random_fss.solve(sampler_kwargs={'profile': 'defaults'})

    input_file = 'input/tai20_5.txt'
    file_fss = create_fss_from_file(input_file)
    file_fss.build_model()
    file_fss.solve(sampler_kwargs={'profile': 'defaults'}, time_limit=5)