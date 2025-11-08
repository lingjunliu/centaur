import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if output_shape is provided, all dimensions of input tensor shape should be divisible by corresponding pooling ratios (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] > 0, And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg1_shape"], i) % Select(v["arg3_values"], i) == 0) for i in range(6)]), True)) if n else
          If(v["arg2_length"] > 0, And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg1_shape"], i) % Select(v["arg3_values"], i) == 0) for i in range(6)]), True))
)

def rule_67_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all(isinstance(e, (float, np.floating)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg3_values = Array('arg3_values', IntSort(), RealSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 67
        rule_67(solver, {'arg1_shape': arg1_shape, 'arg2_length': arg2_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_shape': arg1['shape'], 'arg2_length': arg2['length'], 'arg3_values': arg3['values']}, neg)
