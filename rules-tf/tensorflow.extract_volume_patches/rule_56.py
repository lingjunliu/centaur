import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input has floating point dtype, then the elements in ksizes and strides must be integers (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), And((And([Implies(i < (v["arg2_length"] - 1 + 1), True) for i in range(6)])), (And([Implies(i < (v["arg3_length"] - 1 + 1), True) for i in range(6)]))), False)) if n else
          If(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), And((And([Implies(i < (v["arg2_length"] - 1 + 1), True) for i in range(6)])), (And([Implies(i < (v["arg3_length"] - 1 + 1), True) for i in range(6)]))), False))
)

def rule_56_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 56
        rule_56(solver, {'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length']}, neg)
