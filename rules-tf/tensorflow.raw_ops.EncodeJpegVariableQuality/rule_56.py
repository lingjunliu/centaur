import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if width is power of 2 and height is power of 2 and quality is high, it is a valid configuration  (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 0) == 2, Select(v["arg1_shape"], 0) == 4), Select(v["arg1_shape"], 0) == 8), Select(v["arg1_shape"], 0) == 16), Select(v["arg1_shape"], 0) == 32), Select(v["arg1_shape"], 0) == 64), Select(v["arg1_shape"], 0) == 128), Select(v["arg1_shape"], 0) == 256), Select(v["arg1_shape"], 0) == 512), Select(v["arg1_shape"], 0) == 1024), Select(v["arg1_shape"], 0) == 2048)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 1) == 2, Select(v["arg1_shape"], 1) == 4), Select(v["arg1_shape"], 1) == 8), Select(v["arg1_shape"], 1) == 16), Select(v["arg1_shape"], 1) == 32), Select(v["arg1_shape"], 1) == 64), Select(v["arg1_shape"], 1) == 128), Select(v["arg1_shape"], 1) == 256), Select(v["arg1_shape"], 1) == 512), Select(v["arg1_shape"], 1) == 1024), Select(v["arg1_shape"], 1) == 2048))), (Select(v["arg2_range"], 1) > 90)), True, True)) if n else
          If(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 0) == 2, Select(v["arg1_shape"], 0) == 4), Select(v["arg1_shape"], 0) == 8), Select(v["arg1_shape"], 0) == 16), Select(v["arg1_shape"], 0) == 32), Select(v["arg1_shape"], 0) == 64), Select(v["arg1_shape"], 0) == 128), Select(v["arg1_shape"], 0) == 256), Select(v["arg1_shape"], 0) == 512), Select(v["arg1_shape"], 0) == 1024), Select(v["arg1_shape"], 0) == 2048)), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_shape"], 1) == 2, Select(v["arg1_shape"], 1) == 4), Select(v["arg1_shape"], 1) == 8), Select(v["arg1_shape"], 1) == 16), Select(v["arg1_shape"], 1) == 32), Select(v["arg1_shape"], 1) == 64), Select(v["arg1_shape"], 1) == 128), Select(v["arg1_shape"], 1) == 256), Select(v["arg1_shape"], 1) == 512), Select(v["arg1_shape"], 1) == 1024), Select(v["arg1_shape"], 1) == 2048))), (Select(v["arg2_range"], 1) > 90)), True, True))
)

def rule_56_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 56
        rule_56(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range']}, neg)
