import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# All parameters are integers and positive, and year within range (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((v["arg1_value"] > 1970), (v["arg1_value"] < 2100)), (v["arg2_value"] > 0)), (v["arg2_value"] < 13)), (v["arg3_value"] > 0)), (v["arg3_value"] < 32))) if n else
          And(And(And(And(And((v["arg1_value"] > 1970), (v["arg1_value"] < 2100)), (v["arg2_value"] > 0)), (v["arg2_value"] < 13)), (v["arg3_value"] > 0)), (v["arg3_value"] < 32)))
)

def rule_61_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 61
        rule_61(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
