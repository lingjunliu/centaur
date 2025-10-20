import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Checking if the shape elements can be divided by 2 and must be a float smaller than 100 and if is a complex type is false (Rule 153)

rule_153 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) % 2 == 0, (And(v["arg2_value"] != 10, v["arg2_value"] != 11)))) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) % 2 == 0, (And(v["arg2_value"] != 10, v["arg2_value"] != 11)))) for i in range(6)]))
)

def rule_153_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 153
        rule_153(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_153(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
