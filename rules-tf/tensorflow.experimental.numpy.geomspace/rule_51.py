import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If num is greater than 1, start and stop must not have opposite signs if the dtypes are integers (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] > 1, (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5))), (And(Select(v["arg1_range"], 0) < 0, Select(v["arg2_range"], 1) > 0)) == False, False)) if n else
          If(And(v["arg3_value"] > 1, (Or(Or(Or(Or(v["arg4_value"] == 1, v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5))), (And(Select(v["arg1_range"], 0) < 0, Select(v["arg2_range"], 1) > 0)) == False, False))
)

def rule_51_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 51
        rule_51(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
