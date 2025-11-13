import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Size tensor needs to be a tensor and have both values greater or equal to 0 and have a shape equal to (2, (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_shape"], 0) == 2)) if n else
          And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_shape"], 0) == 2))
)

def rule_115_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 115
        rule_115(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range']}, neg)
