import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# torch.is_autocast_cpu_enabled returns a boolean, so if an int is >= 0 and less than the len of another list, then function must return either true or false (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] >= 0, v["arg3_value"] < v["arg2_length"]), Or(v["arg1_value"] == True, v["arg1_value"] == False), Or(v["arg1_value"] == True, v["arg1_value"] == False))) if n else
          If(And(v["arg3_value"] >= 0, v["arg3_value"] < v["arg2_length"]), Or(v["arg1_value"] == True, v["arg1_value"] == False), Or(v["arg1_value"] == True, v["arg1_value"] == False)))
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, str) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 46
        rule_46(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
