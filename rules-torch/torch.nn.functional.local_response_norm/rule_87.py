import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# k should be large enough in comparison to alpha and the input range to avoid extremely small values during normalization, preventing potential division by zero and also limit the value of max to be within reasonable range (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(And(And(Select(v["arg1_range"], 1) < 1000, Select(v["arg1_range"], 0) > (0 - 1000)), v["arg3_value"] > (Select(v["arg1_range"], 1) + 1000) * v["arg2_value"])) if n else
          And(And(Select(v["arg1_range"], 1) < 1000, Select(v["arg1_range"], 0) > (0 - 1000)), v["arg3_value"] > (Select(v["arg1_range"], 1) + 1000) * v["arg2_value"]))
)

def rule_87_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 87
        rule_87(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
