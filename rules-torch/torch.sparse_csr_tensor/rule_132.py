import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If requires_grad is true, all input tensors should be float and have ndim less than 4 and has to have all elements greater than 0 and all elements have to be less than 100 and has to be divisible by 3. (Rule 132)

rule_132 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"], And(And(And(And(And(And(And(And(And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10))), (Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), (v["arg1_ndim"] < 4)), (v["arg2_ndim"] < 4)), (v["arg3_ndim"] < 4)), (Select(v["arg1_range"], 0) > 0)), (Select(v["arg2_range"], 0) > 0)), (Select(v["arg3_range"], 0) > 0)), (Select(v["arg1_range"], 1) < 100)), (Select(v["arg2_range"], 1) < 100)), (Select(v["arg3_range"], 1) < 100)), (Select(v["arg1_range"], 0) % 3 == 0)), (Select(v["arg2_range"], 0) % 3 == 0)), (Select(v["arg3_range"], 0) % 3 == 0)), True)) if n else
          If(v["arg4_value"], And(And(And(And(And(And(And(And(And(And(And(And(And(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Or(Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10))), (Or(Or(Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10))), (v["arg1_ndim"] < 4)), (v["arg2_ndim"] < 4)), (v["arg3_ndim"] < 4)), (Select(v["arg1_range"], 0) > 0)), (Select(v["arg2_range"], 0) > 0)), (Select(v["arg3_range"], 0) > 0)), (Select(v["arg1_range"], 1) < 100)), (Select(v["arg2_range"], 1) < 100)), (Select(v["arg3_range"], 1) < 100)), (Select(v["arg1_range"], 0) % 3 == 0)), (Select(v["arg2_range"], 0) % 3 == 0)), (Select(v["arg3_range"], 0) % 3 == 0)), True))
)

def rule_132_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_value == arg4)

        # Constraints for rule 132
        rule_132(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_132(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
