import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# pooling ratios height and width must be different to 1.0, if the overlapping is enabled (Rule 122)

rule_122 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(Select(v["arg2_values"], 1) != 1.0, Select(v["arg2_values"], 2) != 1.0), False)) if n else
          If(v["arg1_value"] == True, And(Select(v["arg2_values"], 1) != 1.0, Select(v["arg2_values"], 2) != 1.0), False))
)

def rule_122_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 122
        rule_122(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_122(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values']}, neg)
