import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If there exists at least one non-empty string in the input, and padding is disabled, ensure each string length is a multiple of 3 for efficient base64 encoding (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(And((Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), v["arg2_value"] == False), And([Implies(j < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], j) % 3 == 0) for j in range(6)]), True)) if n else
          If(And((Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), v["arg2_value"] == False), And([Implies(j < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], j) % 3 == 0) for j in range(6)]), True))
)

def rule_59_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 59
        rule_59(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
