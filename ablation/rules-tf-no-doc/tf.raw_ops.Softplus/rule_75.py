import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Beta can be used to adjust the sharpness of the transition of the Softplus function. If beta is too close to zero, the transition may be too flat to be useful. If input has negative values, it is undesirable. (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_range"], 0) < 0, v["arg1_value"] > 0.00001, True)) if n else
          If(Select(v["arg2_range"], 0) < 0, v["arg1_value"] > 0.00001, True))
)

def rule_75_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 75
        rule_75(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
