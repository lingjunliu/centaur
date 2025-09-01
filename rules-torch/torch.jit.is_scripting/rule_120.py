import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Compilation validation of certain tensor and requires to be in certain validation point and has all positive in all d (Rule 120)

rule_120 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(Select(v["arg2_range"], 0) > 0, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)])), True)) if n else
          If(v["arg1_value"] == True, And(Select(v["arg2_range"], 0) > 0, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)])), True))
)

def rule_120_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 120
        rule_120(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_120(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
