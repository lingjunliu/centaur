import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If tensor 'x' is of complex type, then all elements of tensor 'p' also have to be of a complex type (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(If((And(9 <= v["arg2_dtype"], v["arg2_dtype"] <= 10)), And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))) for i in range(6)]), True)) if n else
          If((And(9 <= v["arg2_dtype"], v["arg2_dtype"] <= 10)), And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))) for i in range(6)]), True))
)

def rule_96_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 96
        rule_96(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype']}, neg)
