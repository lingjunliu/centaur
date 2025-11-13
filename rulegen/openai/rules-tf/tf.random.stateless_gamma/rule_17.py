import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# shape list must be non-empty, non-negative, and its last dim matches 1-D alpha/beta list length (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_length"] >= 1, v["arg2_length"] == v["arg3_length"]), Select(v["arg1_values"], v["arg1_length"] - 1) == v["arg2_length"]), And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) >= 0) for i in range(6)]))) if n else
          And(And(And(v["arg1_length"] >= 1, v["arg2_length"] == v["arg3_length"]), Select(v["arg1_values"], v["arg1_length"] - 1) == v["arg2_length"]), And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) >= 0) for i in range(6)])))
)

def rule_17_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all(isinstance(e, (float, np.floating)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 17
        rule_17(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_length': arg2_length, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length']}, neg)
