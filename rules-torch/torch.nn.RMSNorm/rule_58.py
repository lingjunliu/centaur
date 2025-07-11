import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Storage size cannot overflow (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) < 200, Select(v["arg1_shape"], 1) < 200), Select(v["arg1_shape"], 2) < 200), Select(v["arg1_shape"], 3) < 200), Select(v["arg1_shape"], 4) < 200), Select(v["arg1_shape"], 5) < 200), Select(v["arg1_shape"], 6) < 200), Select(v["arg1_shape"], 7) < 200), Select(v["arg1_shape"], 8) < 200)) if n else
          And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) < 200, Select(v["arg1_shape"], 1) < 200), Select(v["arg1_shape"], 2) < 200), Select(v["arg1_shape"], 3) < 200), Select(v["arg1_shape"], 4) < 200), Select(v["arg1_shape"], 5) < 200), Select(v["arg1_shape"], 6) < 200), Select(v["arg1_shape"], 7) < 200), Select(v["arg1_shape"], 8) < 200))
)

def rule_58_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 58
        rule_58(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_shape': arg1['shape']}, neg)
