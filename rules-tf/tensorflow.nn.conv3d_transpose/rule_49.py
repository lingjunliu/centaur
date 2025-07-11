import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If strides is a list of length 5 and dilation is a list of length 5, then strides[i] must be greater or equal to dilation[i], for i = 1,2,3 (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] == 5, v["arg2_length"] == 5), And(And(Select(v["arg1_values"], 1) >= Select(v["arg2_values"], 1), Select(v["arg1_values"], 2) >= Select(v["arg2_values"], 2)), Select(v["arg1_values"], 3) >= Select(v["arg2_values"], 3)), False)) if n else
          If(And(v["arg1_length"] == 5, v["arg2_length"] == 5), And(And(Select(v["arg1_values"], 1) >= Select(v["arg2_values"], 1), Select(v["arg1_values"], 2) >= Select(v["arg2_values"], 2)), Select(v["arg1_values"], 3) >= Select(v["arg2_values"], 3)), False))
)

def rule_49_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 49
        rule_49(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
