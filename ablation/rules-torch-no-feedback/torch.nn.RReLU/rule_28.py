import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if input tensor is almost zero and upper is negative and lower is negative and inplace is True it can potentially produce inf, so that should be avoided (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And((And(Select(v["arg4_range"], 1) < 0.00001, Select(v["arg4_range"], 0) > -0.00001)), (v["arg1_value"] < 0)), (v["arg2_value"] < 0)), v["arg3_value"]), False, True)) if n else
          If(And(And(And((And(Select(v["arg4_range"], 1) < 0.00001, Select(v["arg4_range"], 0) > -0.00001)), (v["arg1_value"] < 0)), (v["arg2_value"] < 0)), v["arg3_value"]), False, True))
)

def rule_28_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_range': arg4_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_range': arg4['range']}, neg)
