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

import typing

import numpy as np
import numpy.typing

from dwave.optimization.mathematical import add, maximum, minimum
from dwave.optimization.model import Model
from dwave.system import LeapHybridNLSampler

__all__ = ["flow_shop_scheduling" ]


def _2d_nonnegative_int_array(**kwargs: numpy.typing.ArrayLike) -> typing.Iterator[np.ndarray]:
    """Coerce all given array-likes to 2d NumPy arrays of non-negative integers and
    raise a consistent error message if it cannot be cast.

    Keyword argument names must match the argument names of the calling function
    for the error message to make sense.
    """
    for argname, array in kwargs.items():
        try:
            array = np.atleast_2d(np.asarray(array, dtype=int))
        except (ValueError, TypeError):
            raise ValueError(f"`{argname}` must be a 2d array-like of non-negative integers")
        
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"`{argname}` must be a 2d array-like of non-negative integers")

        if array.ndim != 2 or (array < 0).any():
            raise ValueError(f"`{argname}` must be a 2d array-like of non-negative integers")

        yield array


# Dev note: this is currently private as it's not optimized and doesn't do the full
# set of features we'd eventually like
def _from_constrained_quadratic_model(cqm) -> Model:
    """Construct a NL model from a :class:`dimod.ConstrainedQuadraticModel`."""

    for v in cqm.variables:
        if cqm.vartype(v).name != "BINARY":
            raise ValueError("CQM must only have binary variables")

    model = Model()

    if not cqm.num_variables():
        return model

    x = model.binary(cqm.num_variables())

    def quadratic_model(qm):
        # Get the indices of the variables in the QM
        # In the future we could test if this is a range and do basic indexing in that case
        indices = model.constant([cqm.variables.index(v) for v in qm.variables])

        # Get the biases in COO format
        # We should add `to_numpy_vectors()` to dimod objective/constraints
        linear = model.constant([qm.get_linear(v) for v in qm.variables])

        if qm.is_linear():
            out = (linear * x[indices]).sum()
        else:
            irow = []
            icol = []
            quadratic = []
            for u, v, bias in qm.iter_quadratic():
                irow.append(qm.variables.index(u))
                icol.append(qm.variables.index(v))
                quadratic.append(bias)

            out = model.quadratic_model(x[indices], (np.asarray(quadratic), (irow, icol)), linear)

        if qm.offset:
            return out + model.constant(qm.offset)

        return out

    # Minimize the objective
    model.minimize(quadratic_model(cqm.objective))

    for comp in cqm.constraints.values():
        lhs = quadratic_model(comp.lhs)
        rhs = model.constant(comp.rhs)

        if comp.sense.value == "==":
            model.add_constraint(lhs == rhs)
        elif comp.sense.value == "<=":
            model.add_constraint(lhs <= rhs)
        elif comp.sense.value == ">=":
            model.add_constraint(rhs <= lhs)
        else:
            raise RuntimeError("unexpected sense")

    return model


def flow_shop_scheduling(processing_times: numpy.typing.ArrayLike
                                            ) -> typing.Tuple[Model, list[list[int]]]:
    r"""Generate a model encoding a flow-shop scheduling problem.

    `Flow-shop scheduling <https://en.wikipedia.org/wiki/Flow-shop_scheduling>`_ 
    is a variant of the renowned :func:`.job_shop_scheduling` optimization problem. 
    Given `n` jobs to schedule on `m` machines, with specified processing 
    times for each job per machine, minimize the makespan (the total 
    length of the schedule for processing all the jobs). For every job, the 
    `i`-th operation is executed on the `i`-th machine. No machine can 
    perform more than one operation simultaneously. 
   
    `E. Taillard <http://mistic.heig-vd.ch/taillard/problemes.dir/problemes.html>`_
    provides benchmark instances compatible with this generator.

    .. Note::
        There are many ways to model flow-shop scheduling. The model returned
        by this function may or may not give the best performance for your
        problem.

    Args:
        processing_times:
            Processing times, as an :math:`n \times m` |array-like|_ of 
            integers, where ``processing_times[n, m]`` is the time job 
            `n` is on machine `m`.
        
    Returns:
        A model encoding the flow-shop scheduling problem and the list of
        end-times for the problem solution.
        
    Examples:
    
        This example creates a model for a flow-shop-scheduling problem
        with two jobs on three machines. For example, the second job 
        requires processing for 20 time units on the first machine in 
        the flow of operations.  
    
        >>> from dwave.optimization.generators import flow_shop_scheduling
        ...
        >>> processing_times = [[10, 5, 7], [20, 10, 15]]
        >>> model, end_times = flow_shop_scheduling(processing_times=processing_times)

    """
    import pdb
    pdb.set_trace()
    processing_times = next(_2d_nonnegative_int_array(processing_times=processing_times))
    if not processing_times.size:
        raise ValueError("`processing_times` must not be empty")

    num_machines, num_jobs = processing_times.shape

    model = Model()
    
    # Add the constant processing-times matrix
    times = model.constant(processing_times)
    
    # The decision symbol is a num_jobs-long array of integer variables
    order = model.list(num_jobs)

    end_times = []
    for machine_m in range(num_machines):

        machine_m_times = []
        if machine_m == 0:

            for job_j in range(num_jobs):
            
                if job_j == 0:
                    machine_m_times.append(times[machine_m, :][order[job_j]])
                else:
                    end_job_j = times[machine_m, :][order[job_j]]
                    end_job_j += machine_m_times[-1]
                    machine_m_times.append(end_job_j)

        else:

            for job_j in range(num_jobs):
            
                if job_j == 0:
                    end_job_j = end_times[machine_m - 1][job_j]
                    end_job_j += times[machine_m, :][order[job_j]]
                    machine_m_times.append(end_job_j)
                else:
                    end_job_j = maximum(end_times[machine_m - 1][job_j], machine_m_times[-1])
                    end_job_j += times[machine_m, :][order[job_j]]
                    machine_m_times.append(end_job_j)

        end_times.append(machine_m_times)

    makespan = end_times[-1][-1]
    
    # The objective is to minimize the last end time
    model.minimize(makespan)

    model.lock()
    
    return model, end_times


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


class FlowShopScheduler:
    '''
    This class is used to generate a model encoding a flow-shop scheduling problem.
    '''

    def __init__(self, machines: list[Machine], jobs: list[Job], cooldown_periods: list[CooldownPeriod]=[]):
        """Initialize a flow-shop scheduler.

        Args:
            machines (list[Machine]): A list of machines to execute jobs. The order of this list
                indicates the order all jobs must follow.
            jobs (list[Job]): A list of jobs to be executed on the machines. The order of this list
                is inconsequential.
            cooldown_periods (list[CooldownPeriod], optional): A list of
                cooldown periods associated with specific machines. Defaults to []. This order of
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
        for cooldown_period in self.cooldown_periods:
            if cooldown_period.machine not in self.machines:
                raise ValueError("Cooldown period machine not in machines list")
            machine_idx = self.machines.index(cooldown_period.machine)
            self.ordering_groups.append(machines[last_idx:machine_idx+1])
            last_idx = machine_idx + 1
        self.ordering_groups.append(machines[last_idx:])



    def build_model(self):
        """Build the flow-shop scheduling model.

        Returns:
            Tuple[Model, List[ArrayObserver]]: A model encoding the flow-shop
            scheduling problem and the list of end-times for the problem solution.
        """

        processing_times = [[job.processing_times[machine] for job in self.jobs] for machine in self.machines]
        processing_times = next(_2d_nonnegative_int_array(processing_times=processing_times))


        self.model = Model()
        
        # Add the constant processing-times matrix
        times = self.model.constant(processing_times)

        cooldown_times = [[x.durations[job] for job in self.jobs] for x in self.cooldown_periods]
        self.cooldown_times = self.model.constant(cooldown_times)
        

        num_jobs = len(self.jobs)
        self.orders = [self.model.list(num_jobs) for x in range(len(self.ordering_groups))]
        self.order_group_ends = [self.model.integer(shape=num_jobs, lower_bound=0, upper_bound=1000) for x in range(len(self.ordering_groups)-1)]

        self.end_times = []
        for order_idx, machines in enumerate(self.ordering_groups):

            order = self.orders[order_idx]

            machine_m = self.machines.index(machines[0])
            machine_m_times = []
            if order_idx == 0:

                for job_j, job in enumerate(self.jobs):
                
                    if job_j == 0:
                        machine_m_times.append(times[machine_m, :][order[job_j]])
                    else:
                        end_job_j = times[machine_m][order[job_j]]
                        end_job_j += machine_m_times[-1]
                        machine_m_times.append(end_job_j)
                    self.model.add_constraint(self.order_group_ends[order_idx][order[job_j]] == \
                                              machine_m_times[-1] + self.cooldown_times[order_idx][order[job_j]])
                    
            else:

                for job_j, job in enumerate(self.jobs): #we'll go in order of self.order2
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

            self.end_times.append(machine_m_times)

        makespan = maximum(self.end_times[-1])

        # The objective is to minimize the last end time
        self.model.minimize(makespan)
        self.model.lock()


    def _calculate_end_times(self) -> list[list[int]]:
        """Calculate the end-times for the FSS job results.

        Helper function to calculate the end-times for the FSS job
        results obtained from the NL Solver. Taken directly from the
        FSS generator in the NL Solver generators module.

        Update when symbol labels are supported.

        Returns:
            list[list[int]]: end-times from the problem results
        """
        # times = self.processing_times

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
                        job_last_machine_end = self.job_schedules[job][-1][-1] + self.job_schedules[job][-1][-2]
                        start_time = job_last_machine_end
                    else:
                        if len(self.job_schedules[job]) == 0:
                            job_last_machine_end = 0
                        else:
                            job_last_machine_end = self.job_schedules[job][-1][-1] + self.job_schedules[job][-1][-2]
                        machine_last_job_end = self.machine_schedules[machine][-1][-2]
                        start_time = np.maximum(job_last_machine_end, machine_last_job_end)

                    end_time = start_time + job.processing_times[machine]
                    if machine_idx == len(self.ordering_groups[order_idx]) - 1:
                        cooldown_time = self.cooldown_periods[order_idx-1].durations[job]
                    else:
                        cooldown_time = 0
                    self.job_schedules[job].append((start_time, end_time, cooldown_time))
                    self.machine_schedules[machine].append((job, start_time, end_time, cooldown_time))
       


    def solve(self, time_limit: int=None, sampler_kwargs: dict={}) -> None:
        """Solve the flow-shop scheduling problem.

        Args:
            time_limit (int, optional): Time limit in seconds. Defaults to None.
            **sampler_kwargs: Additional keyword arguments for the sampler.

        """
        sampler = LeapHybridNLSampler(**sampler_kwargs)
        sampler.sample(self.model, time_limit=time_limit, label="FSS demo")

        self._calculate_end_times()
        self.solution = {}
        for machine_idx, machine_times in enumerate(self.end_times):
            for job_idx, end_time in zip(self.job_order, machine_times):

                duration = self.processing_times[machine_idx][job_idx]
                self.solution[(job_idx, machine_idx)] = end_time - duration, duration


    def __repr__(self):
        return f"FlowShopScheduler(processing_times={self.processing_times})"

    def __str__(self):
        return f"FlowShopScheduler with processing times:\n{self.processing_times}"
    

if __name__ == '__main__':
    machines = [Machine(0), Machine(1)]
    jobs = [Job(0, {machines[0]: 1, machines[1]:5}), Job(1, {machines[0]: 4, machines[1]:5}), Job(2, {machines[0]: 1, machines[1]:4})]
    cooldown_periods = [CooldownPeriod(machines[0], {jobs[0]: 5, jobs[1]: 30, jobs[2]: 30})]
    fss = FlowShopScheduler(machines, jobs, cooldown_periods)
    # fss = FlowShopScheduler(processing_times=[[1,4,1], [5, 5, 4]], cooldown_times=[5,30,30])
    fss.build_model()
    fss.solve(sampler_kwargs={'profile': 'defaults'})