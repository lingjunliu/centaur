import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If use_nesterov is false and lr is greater than 0, the max absolute value of grad should be a reasonable value. (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, Select(v["arg2_range"], 1) > 0), Select(v["arg2_range"], 1) < 1000, False)) if n else
          If(And(v["arg1_value"] == False, Select(v["arg2_range"], 1) > 0), Select(v["arg2_range"], 1) < 1000, False))
)

def rule_104_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 104
        rule_104(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
