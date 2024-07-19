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

from dwave.optimization.mathematical import add, maximum 
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


class FlowShopScheduler:
    '''
    This class is used to generate a model encoding a flow-shop scheduling problem.
    '''

    def __init__(self, processing_times: numpy.typing.ArrayLike):
        """Initialize a flow-shop scheduler with the given processing

        Args:
            processing_times (numpy.typing.ArrayLike): Processing times, as 
            an :math:`n \times m` |array-like|_ of integers, where
            ``processing_times[n, m]`` is the time job `n` is on machine `m`.
        
        Raises:
            ValueError: If `processing_times` is empty.
        """
        if len(processing_times) == 0:
            raise ValueError("`processing_times` must not be empty")
        self.processing_times = processing_times


    def build_model(self):
        """Build the flow-shop scheduling model.

        Returns:
            Tuple[Model, List[ArrayObserver]]: A model encoding the flow-shop
            scheduling problem and the list of end-times for the problem solution.
        """
        processing_times = next(_2d_nonnegative_int_array(processing_times=self.processing_times))

        num_machines, num_jobs = processing_times.shape
        self.machines = list(range(num_machines))
        self.jobs = list(range(num_jobs))

        self.model = Model()
        
        # Add the constant processing-times matrix
        times = self.model.constant(processing_times)
        
        # The decision symbol is a num_jobs-long array of integer variables
        order = self.model.list(num_jobs)

        self.end_times = []
        for machine_m in self.machines:

            machine_m_times = []
            if machine_m == 0:

                for job_j in self.jobs:
                
                    if job_j == 0:
                        machine_m_times.append(times[machine_m, :][order[job_j]])
                    else:
                        end_job_j = times[machine_m, :][order[job_j]]
                        end_job_j += machine_m_times[-1]
                        machine_m_times.append(end_job_j)

            else:

                for job_j in self.jobs:
                
                    if job_j == 0:
                        end_job_j = self.end_times[machine_m - 1][job_j]
                        end_job_j += times[machine_m, :][order[job_j]]
                        machine_m_times.append(end_job_j)
                    else:
                        end_job_j = maximum(self.end_times[machine_m - 1][job_j], machine_m_times[-1])
                        end_job_j += times[machine_m, :][order[job_j]]
                        machine_m_times.append(end_job_j)

            self.end_times.append(machine_m_times)
        makespan = self.end_times[-1][-1]
        
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
        times = self.processing_times

        self.job_order = [x for x in next(self.model.iter_decisions()).state(0).astype(int)]

        self.end_times = []
        for machine_m, _ in enumerate(times):

            machine_m_times = []
            if machine_m == 0:

                for job_j, order_j in enumerate(self.job_order):

                    if job_j == 0:
                        machine_m_times.append(times[machine_m][order_j])
                    else:
                        end_job_j = times[machine_m][order_j]
                        end_job_j += machine_m_times[-1]
                        machine_m_times.append(end_job_j)

            else:

                for job_j, order_j in enumerate(self.job_order):

                    if job_j == 0:
                        end_job_j = self.end_times[machine_m - 1][job_j]
                        end_job_j += times[machine_m][order_j]
                        machine_m_times.append(end_job_j)
                    else:
                        end_job_j = max(self.end_times[machine_m - 1][job_j], machine_m_times[-1])
                        end_job_j += times[machine_m][order_j]
                        machine_m_times.append(end_job_j)

            self.end_times.append(machine_m_times)


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
    fss = FlowShopScheduler(processing_times=[[10, 5, 7], [20, 10, 15]])
    fss.build_model()
    fss.solve(sampler_kwargs={'profile': 'defaults'})
