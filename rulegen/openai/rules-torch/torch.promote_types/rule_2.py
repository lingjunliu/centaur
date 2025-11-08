import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# neither input may be a string or numpy dtype sentinel code [11, 12] (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (12 + 1), (And(v["arg1_value"] != i, v["arg2_value"] != i))) for i in range(6)])) if n else
          And([Implies(i < (12 + 1), (And(v["arg1_value"] != i, v["arg2_value"] != i))) for i in range(6)]))
)

def rule_2_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 2
        rule_2(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
