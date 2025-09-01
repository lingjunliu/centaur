import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# crow_indices' last value must be less or equal than num blocks specified by size if size is given and size should have the expected shape. It also checks if size matches crow and col indices if specified (Rule 117)

rule_117 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, And(Select(v["arg2_shape"], 0) <= Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1), Select(v["arg3_shape"], 0) <= Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1)), True)) if n else
          If(v["arg1_length"] > 0, And(Select(v["arg2_shape"], 0) <= Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1), Select(v["arg3_shape"], 0) <= Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1)), True))
)

def rule_117_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 117
        rule_117(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_117(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape']}, neg)
