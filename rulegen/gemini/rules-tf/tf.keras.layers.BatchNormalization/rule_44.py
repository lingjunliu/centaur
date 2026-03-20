import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If renorm is True, then both renorm_clipping rmax and rmin should be defined and non-negative (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], And(And(v["arg2_length"] == 3, Select(v["arg2_values"], 0) >= 0), Select(v["arg2_values"], 1) >= 0), True)) if n else
          If(v["arg1_value"], And(And(v["arg2_length"] == 3, Select(v["arg2_values"], 0) >= 0), Select(v["arg2_values"], 1) >= 0), True))
)

def rule_44_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, tuple) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 44
        rule_44(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
