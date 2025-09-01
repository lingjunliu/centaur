import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# v_1 must possess the required _forward_pre_hooks attributes. In other words, it must be of type Module or something equivalent that won't throw an AttributeError when accessing its internal methods and properties. (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_range"], 0) > -10000) if n else
          Select(v["arg1_range"], 0) > -10000)
)

def rule_56_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 56
        rule_56(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_range': arg1['range']}, neg)
