import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Both inputs must be tensors of floating point type and not complex or numpy arrays, and it is not allowed to have both tensors with all elements equal to 0 to avoid getting a NaN as a result in the output (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(And(And((And(And(And(And(v["arg1_dtype"] != 10, v["arg1_dtype"] != 11), v["arg1_dtype"] != 12), 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 9)), (And(And(And(And(v["arg2_dtype"] != 10, v["arg2_dtype"] != 11), v["arg2_dtype"] != 12), 6 <= v["arg2_dtype"]), v["arg2_dtype"] <= 9))), ((And(Select(v["arg1_range"], 1) == 0, Select(v["arg2_range"], 1) == 0)) == False))) if n else
          And(And((And(And(And(And(v["arg1_dtype"] != 10, v["arg1_dtype"] != 11), v["arg1_dtype"] != 12), 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 9)), (And(And(And(And(v["arg2_dtype"] != 10, v["arg2_dtype"] != 11), v["arg2_dtype"] != 12), 6 <= v["arg2_dtype"]), v["arg2_dtype"] <= 9))), ((And(Select(v["arg1_range"], 1) == 0, Select(v["arg2_range"], 1) == 0)) == False)))
)

def rule_75_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 75
        rule_75(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
