import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Filter tensor's shape should be positive (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (2 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])) if n else
          And([Implies(i < (2 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))
)

def rule_51_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 51
        rule_51(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_shape': arg1['shape']}, neg)
