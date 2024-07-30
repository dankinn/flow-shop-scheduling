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
    def __init__(self, id_: int, processing_times: dict[int], hard_deadline: int=None):
        """Initialize a job.

        Args:
            id_ (int): The job ID, a unique identifier to distinguish
                between jobs.
            processing_times (dict[int]): A dictionary of processing times
                for each machine, in the form {machine: processing_time}.
            hard_deadline (int, optional): A hard deadline for the job. Defaults 
                to None.

        """
        self.id_ = id_
        self.processing_times = processing_times
        self.hard_deadline = hard_deadline

    def __repr__(self):
        return f"Job {self.id_}"

    def __str__(self):
        return f"Job {self.id_}"
    
    def __eq__(self, other: object):
        if not isinstance(other, Job):
            return False
        return self.id_ == other.id_
    
    def __hash__(self) -> int:
        return hash('job_' + str(self.id_))
    

class Machine:
    """A machine object to execute jobs."""
    def __init__(self, id_: str, maintenance_interval: int=None, maintenance_time: int=0):
        self.id = id_
        self.maintenance_interval = maintenance_interval
        self.maintenance_time = maintenance_time

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
    """A cooldown period associated with a specific Machine.
    The cooldown period occurs after a job is executed on the
    machine.
    """
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
    

class ScheduledMaintenance:
    """A scheduled maintenance period for a machine."""
    def __init__(self, machine: Machine, start_time: int, end_time: int):
        """Initialize a scheduled maintenance period.

        Args:
            machine (Machine): The machine to be maintained.
            start_time (int): The start time of the maintenance period.
            end_time (int): The end time of the maintenance period.
        """
        self.machine = machine
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time

    def __repr__(self):
        return f"Scheduled maintenance: {self.machine.id} ({self.start_time}, {self.end_time})"

    def __str__(self):
        return f"Scheduled maintenance: {self.machine.id} ({self.start_time}, {self.end_time})"
    
    def __eq__(self, other: object):
        if not isinstance(other, ScheduledMaintenance):
            return False
        return self.machine == other.machine \
                and self.start_time == other.start_time \
                and self.end_time == other.end_time
    
    def __hash__(self) -> int:
        return hash('scheduled_maintenance_' + str(self.machine.id))

    
class ScheduledJob:
    """A scheduled job with start and end times."""
    def __init__(self, job: Job,
                 start_time: int,
                 end_time: int,
                 cooldown_time: int=None):
        """Initialize a scheduled job.

        Args:
            job (Job): The job to be scheduled.
            start_time (int): The start time of the job.
            end_time (int): The end time of the job.
            cooldown_time (int, optional): The cooldown time after the job is
                executed. Defaults to None.
        """
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
    
    def __eq__(self, other: object):
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
                 cooldown_periods: list[CooldownPeriod]=[],
                 use_hard_deadline: bool=False):
        """Initialize a flow-shop scheduler.

        Args:
            machines (list[Machine]): A list of machines to execute jobs. The order of this list
                indicates the order all jobs must follow.
            jobs (list[Job]): A list of jobs to be executed on the machines. The order of this list
                is inconsequential.
            cooldown_periods (list[CooldownPeriod], optional): A list of
                cooldown periods associated with specific machines. Defaults to []. The order of
                this list is inconsequential.
            use_hard_deadline (bool, optional): A flag to indicate whether to use hard deadlines
                for the jobs. Defaults to False. If True, then any job that has a 
                hard_deadline attribute will have a constraint added to ensure that the job
                is completed before the deadline.
        """
        if len(machines) == 0:
            raise ValueError("At least one machine is required")
        if len(jobs) == 0:
            raise ValueError("At least one job is required")
        self.machines = machines
        self.jobs = jobs
        self.cooldown_periods = cooldown_periods
        self.use_hard_deadline = use_hard_deadline

        self.order_groups = []
        last_idx = 0
        self.cooldown_periods.sort(key=lambda x: self.machines.index(x.machine))
        for cooldown_period in self.cooldown_periods:
            if cooldown_period.machine not in self.machines:
                raise ValueError("Cooldown period machine not in machines list")
            machine_idx = self.machines.index(cooldown_period.machine)
            self.order_groups.append(machines[last_idx:machine_idx+1])
            last_idx = machine_idx + 1
        if last_idx < len(machines):
            self.order_groups.append(machines[last_idx:])


    def generate_integer_bounds(self) -> list[list[int]]:
        """Generate integer bounds for the flow-shop scheduling problem.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of 2D arrays representing the integer
                bounds for the flow-shop scheduling problem.
        """
        ordering_group_bounds = [[None, None] for _ in range(len(self.order_groups))]
        min_time = 0
        total_time = 0
        for idx, ordering_group in enumerate(self.order_groups):
            ordering_group_bounds[idx][0] = min_time
            min_time_nominee = float('inf')
            for job in self.jobs:
                job_time = 0
                for machine in ordering_group:
                    job_time += job.processing_times[machine]
                if idx < len(self.cooldown_periods):
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
        self.orders = [self.model.list(num_jobs) for _ in range(len(self.order_groups))]
        ordering_group_bounds = self.generate_integer_bounds()
        self.order_group_ends = [self.model.integer(shape=num_jobs, lower_bound=lb, upper_bound=ub) \
                                 for [lb, ub] in ordering_group_bounds]

        self.end_times = []
        for order_idx, machines in enumerate(self.order_groups):
            order = self.orders[order_idx]
            order_end_times = []
            for machine_idx in range(len(machines)):
                machine_m = self.machines.index(machines[machine_idx])
                maintenance_interval = machines[machine_idx].maintenance_interval
                machine_m_times = []
                if order_idx == 0: # the first order is special because we don't need to consider the prior order
                    for job_j in range(len(self.jobs)): #iterate through the order of the first grouping
                        if maintenance_interval is not None and (job_j + 1) % maintenance_interval == 0:
                            maintenance_time = self.model.constant(machines[machine_idx].maintenance_time)
                        else:
                            maintenance_time = self.model.constant(0)
                        if machine_idx == 0:
                            if job_j == 0: #if a job is the first in order, then it'll start at time 0 
                                # and end after its processing time
                                end_time = times[machine_m, :][order[job_j]] + maintenance_time
                                machine_m_times.append(end_time)
                            else: # otherwise, the job will start at the prior job's end time
                                end_job_j = times[machine_m][order[job_j]]
                                end_job_j += machine_m_times[-1]
                                end_job_j += maintenance_time
                                machine_m_times.append(end_job_j)
                        else:
                            if job_j == 0:
                                end_job_j = order_end_times[machine_m - 1][job_j]
                                end_job_j += times[machine_m, :][order[job_j]]
                                end_job_j += maintenance_time
                                machine_m_times.append(end_job_j)
                            else:
                                end_job_j = maximum(order_end_times[machine_m - 1][job_j], machine_m_times[-1])
                                end_job_j += times[machine_m, :][order[job_j]]
                                end_job_j += maintenance_time
                                machine_m_times.append(end_job_j)
                        
                
                        if (machine_idx == len(machines) - 1) and (len(self.order_groups) >= 1): 
                            #if there are cooldown periods, we'll add the cooldown time
                            end_time = machine_m_times[-1]
                            if len(self.cooldown_periods) >= order_idx + 1:
                                end_time += cooldown_times[order_idx][order[job_j]]
                            end_time += maintenance_time
                            self.model.add_constraint(self.order_group_ends[order_idx][order[job_j]] == end_time)
                    
                else: #if we're not on the first ordering group, we must consider the prior ordering group's end times
                    for job_j in range(len(self.jobs)): #iterate through this groups ordering
                        if maintenance_interval is not None and (job_j + 1) % maintenance_interval == 0:
                            maintenance_time = self.model.constant(machines[machine_idx].maintenance_time)
                        else:
                            maintenance_time = self.model.constant(0)
                        if machine_idx == 0:
                            if job_j == 0:
                                end_job_j = self.order_group_ends[order_idx-1][order[job_j]]
                                end_job_j += times[machine_m][order[job_j]]
                                end_job_j += maintenance_time
                                machine_m_times.append(end_job_j)
                            else:
                                end_job_j = maximum(self.order_group_ends[order_idx-1][order[job_j]],
                                                        machine_m_times[-1],
                                                        )
                                end_job_j += times[machine_m][order[job_j]]
                                end_job_j += maintenance_time
                                machine_m_times.append(end_job_j)
                        else:
                            if job_j == 0:
                                end_job_j = order_end_times[-1][job_j]
                                end_job_j += times[machine_m][order[job_j]]
                                end_job_j += maintenance_time
                                machine_m_times.append(end_job_j)
                            else:
                                end_job_j = maximum(self.order_group_ends[order_idx-1][order[job_j]],
                                                        machine_m_times[-1],
                                                        )
                                end_job_j += times[machine_m][order[job_j]]
                                end_job_j += maintenance_time
                                machine_m_times.append(end_job_j)

                        if (machine_idx == len(machines) - 1) and (len(self.order_groups) >= order_idx + 1):
                            #if this is the last machine in this ordering group, we'll add the cooldown time
                            end_time = machine_m_times[-1]
                            if len(self.cooldown_periods) >= order_idx + 1:
                                end_time += cooldown_times[order_idx][order[job_j]]
                            end_time += maintenance_time
                            self.model.add_constraint(self.order_group_ends[order_idx][order[job_j]] == end_time)
                order_end_times.append(machine_m_times)
            self.end_times.append(order_end_times)

        if self.use_hard_deadline:
            for job_idx, job in enumerate(self.jobs):
                if job.hard_deadline is not None:
                    self.model.add_constraint(self.order_group_ends[-1][job_idx] <= self.model.constant(job.hard_deadline))



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
                for machine_idx, machine in enumerate(self.order_groups[order_idx]):
                    maintenance_interval = machine.maintenance_interval 

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
                    if (machine_idx == len(self.order_groups[order_idx]) - 1) and \
                        (order_idx < len(self.order_groups) - 1):
                        cooldown_time = self.cooldown_periods[order_idx].durations[job]
                    else:
                        cooldown_time = 0

                    scheduled_job = ScheduledJob(job, start_time, end_time, cooldown_time)
                    self.job_schedules[job].append(scheduled_job)
                    self.machine_schedules[machine].append(scheduled_job)
                    if maintenance_interval is not None and (group_idx + 1) % maintenance_interval == 0:
                        maintenance_time = machine.maintenance_time
                        maintenance_period = ScheduledMaintenance(machine, end_time, end_time + maintenance_time)
                        self.machine_schedules[machine].append(maintenance_period)

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

    random_fss = create_random_fss(num_machines=4, 
                                   num_jobs=3,
                                   num_cooldown_periods=2,
                                   max_processing_time=10,
                                   max_cooldown_time=10,
                                   seed=0)
    random_fss.jobs[0].hard_deadline = 40
    random_fss.use_hard_deadline = True
    random_fss.machines[0].maintenance_interval = 2
    random_fss.machines[0].maintenance_time = 5
    random_fss.machines[1].maintenance_interval = 3
    random_fss.machines[1].maintenance_time = 3

    random_fss.build_model()
    random_fss.solve(sampler_kwargs={'profile': 'defaults'}, time_limit=10)


    input_file = 'input/tai20_5.txt'
    # cooldown_times = {2: np.random.randint(1, 10, size=20)}
    cooldown_times = {}
    file_fss = create_fss_from_file(input_file, cooldown_times)
    file_fss.use_hard_deadline = True
    file_fss.jobs[0].hard_deadline = 400 
    file_fss.jobs[1].hard_deadline = 500
    file_fss.jobs[2].hard_deadline = 600
    file_fss.machines[0].maintenance_interval = 2
    file_fss.build_model()
    file_fss.solve(sampler_kwargs={'profile': 'defaults'}, time_limit=30)