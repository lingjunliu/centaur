import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if features tensor has a shape of [1,1], the tensor should contain at least one element (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_shape"], 0) == 1, Select(v["arg1_shape"], 1) == 1), Or(Select(v["arg1_range"], 0) != Select(v["arg1_range"], 1), True), True)) if n else
          If(And(Select(v["arg1_shape"], 0) == 1, Select(v["arg1_shape"], 1) == 1), Or(Select(v["arg1_range"], 0) != Select(v["arg1_range"], 1), True), True))
)

def rule_72_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 72
        rule_72(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range']}, neg)
