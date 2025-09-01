import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combined validation rule (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (If(v["arg1_ndim"] == 0, (And(Select(v["arg1_range"], 0) < 1000, Select(v["arg1_range"], 0) > -1000)), True))), (v["arg1_ndim"] <= 1)), (If(v["arg1_dtype"] == 6, (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 500, True)), If(v["arg1_dtype"] == 7, (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 500, True)), If(v["arg1_dtype"] == 8, (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 800, True)), (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 1000, True)))))))) if n else
          And(And(And((Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (If(v["arg1_ndim"] == 0, (And(Select(v["arg1_range"], 0) < 1000, Select(v["arg1_range"], 0) > -1000)), True))), (v["arg1_ndim"] <= 1)), (If(v["arg1_dtype"] == 6, (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 500, True)), If(v["arg1_dtype"] == 7, (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 500, True)), If(v["arg1_dtype"] == 8, (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 800, True)), (If(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) < 1000, True))))))))
)

def rule_32_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 32
        rule_32(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)
