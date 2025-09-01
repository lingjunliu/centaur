import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the 'out' tensor is provided, its dtype should be able to represent the square root of the 'input' tensor's values without loss of precision or overflow, also making sure that the output tensor's type won't be 'Short' (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), 6 <= v["arg2_dtype"]), v["arg2_dtype"] <= 8), Select(v["arg1_range"], 0) >= 0, If(And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), v["arg1_dtype"] <= v["arg2_dtype"], If(And(9 <= v["arg2_dtype"], v["arg2_dtype"] <= 10), v["arg1_dtype"] <= v["arg2_dtype"], If(v["arg2_dtype"] == 2, False, True))))) if n else
          If(And(And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), 6 <= v["arg2_dtype"]), v["arg2_dtype"] <= 8), Select(v["arg1_range"], 0) >= 0, If(And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), v["arg1_dtype"] <= v["arg2_dtype"], If(And(9 <= v["arg2_dtype"], v["arg2_dtype"] <= 10), v["arg1_dtype"] <= v["arg2_dtype"], If(v["arg2_dtype"] == 2, False, True)))))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
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

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 35
        rule_35(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype']}, neg)
