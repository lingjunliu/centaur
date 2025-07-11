import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dimension should be within the valid range for shape access, using a PRIMVAR to check valid index (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (0 + 1), If(v["arg1_ndim"] > 0, And(i >= 0 - v["arg1_ndim"], i < v["arg1_ndim"]), False)) for i in range(6)])) if n else
          And([Implies(i < (0 + 1), If(v["arg1_ndim"] > 0, And(i >= 0 - v["arg1_ndim"], i < v["arg1_ndim"]), False)) for i in range(6)]))
)

def rule_23_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 23
        rule_23(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_ndim': arg1['ndim']}, neg)
