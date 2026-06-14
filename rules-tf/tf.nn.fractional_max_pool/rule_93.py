import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# pooling_ratio: if length is 1, value[0] must be between 1.0 and 3.0 exclusive (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 1, (And(Select(v["arg1_values"], 0) > 1.0, Select(v["arg1_values"], 0) < 3.0)), True)) if n else
          If(v["arg1_length"] == 1, (And(Select(v["arg1_values"], 0) > 1.0, Select(v["arg1_values"], 0) < 3.0)), True))
)

def rule_93_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 93
        rule_93(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)
