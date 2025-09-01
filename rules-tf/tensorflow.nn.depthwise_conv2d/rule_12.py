import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If any dilation is greater than 1 then all strides must be 1 (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 1) for i in range(6)]), And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], j) == 1) for j in range(6)]), True)) if n else
          If(Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 1) for i in range(6)]), And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], j) == 1) for j in range(6)]), True))
)

def rule_12_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 12
        rule_12(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
