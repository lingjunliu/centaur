import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the layout is specified, the shape must have at least two dimensions. (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 6, v["arg2_value"] != 7), v["arg2_value"] != 8), v["arg2_value"] != 9), v["arg2_value"] != 10), v["arg2_value"] != 11), v["arg2_value"] != 12), v["arg2_value"] != 13), v["arg2_value"] != 14), v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 17), v["arg2_value"] != 18), v["arg2_value"] != 19), v["arg2_value"] != 20), v["arg2_value"] != 21), v["arg2_value"] != 22), v["arg1_length"] >= 2, False)) if n else
          If(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 6, v["arg2_value"] != 7), v["arg2_value"] != 8), v["arg2_value"] != 9), v["arg2_value"] != 10), v["arg2_value"] != 11), v["arg2_value"] != 12), v["arg2_value"] != 13), v["arg2_value"] != 14), v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 17), v["arg2_value"] != 18), v["arg2_value"] != 19), v["arg2_value"] != 20), v["arg2_value"] != 21), v["arg2_value"] != 22), v["arg1_length"] >= 2, False))
)

def rule_28_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 28
        rule_28(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
