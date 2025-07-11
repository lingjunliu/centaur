import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# torch.is_autocast_cpu_enabled: The return is a bool, so if all elements in a list of bools are true, and there exists such list, then that has no bearing on the final return. (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(And((And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) == True) for i in range(6)])), v["arg2_length"] > 0), Or(v["arg1_value"] == True, v["arg1_value"] == False), Or(v["arg1_value"] == True, v["arg1_value"] == False))) if n else
          If(And((And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) == True) for i in range(6)])), v["arg2_length"] > 0), Or(v["arg1_value"] == True, v["arg1_value"] == False), Or(v["arg1_value"] == True, v["arg1_value"] == False)))
)

def rule_19_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, bool) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), BoolSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
