import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 1: default axis = 0

rule_1 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["num_tensors"] > 0,
                v["common_ndim"] >= 0,
                v["common_ndim"] <= MAX_N_DIM,
                And([
                    Implies(i < v["common_ndim"],
                            Select(v["common_shape"], i) >= 0)
                    for i in range(MAX_N_DIM)
                ]),
                v["all_shapes_equal"]
            )
        )
    ) if n else
    s.add(
        And(
            v["num_tensors"] > 0,
            v["common_ndim"] >= 0,
            v["common_ndim"] <= MAX_N_DIM,
            And([
                Implies(i < v["common_ndim"],
                        Select(v["common_shape"], i) >= 0)
                for i in range(MAX_N_DIM)
            ]),
            v["all_shapes_equal"]
        )
    )
)
def rule_1_func(arg1, solver=None, neg=False):
    values = next(iter(arg1.values()))

    if not solver:
        if not isinstance(values, list) or len(values) == 0:
            return False
        if not all(isinstance(v, np.ndarray) for v in values):
            return False

        first = values[0]
        for v in values:
            if v.shape != first.shape:
                return False
            if v.dtype != first.dtype:
                return False

        solver = Solver()

        num_tensors = Int('num_tensors')
        common_ndim = Int('common_ndim')
        common_shape = Array('common_shape', IntSort(), IntSort())
        all_shapes_equal = Bool('all_shapes_equal')

        solver.add(num_tensors == len(values))
        solver.add(common_ndim == first.ndim)
        solver.add(all_shapes_equal == True)

        for i in range(first.ndim):
            solver.add(Select(common_shape, i) == first.shape[i])

        rule_1(solver, {
            "num_tensors": num_tensors,
            "common_ndim": common_ndim,
            "common_shape": common_shape,
            "all_shapes_equal": all_shapes_equal
        })

        return solver.check() == sat

    else:
        rule_1(
            solver,
            {
                "num_tensors": values["num_tensors"],
                "common_ndim": values["common_ndim"],
                "common_shape": values["common_shape"],
                "all_shapes_equal": values["all_shapes_equal"]
            },
            neg
        )