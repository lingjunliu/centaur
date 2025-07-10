import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Check src and tgt for dimension compatibility and sequence length agreement, depending on batch_first (Rule 149)

rule_149 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == False, And(And(v["arg1_ndim"] >= 2, v["arg2_ndim"] >= 2), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), v["arg1_ndim"] >= 2), v["arg2_ndim"] >= 2))) if n else
          If(v["arg3_value"] == False, And(And(v["arg1_ndim"] >= 2, v["arg2_ndim"] >= 2), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), v["arg1_ndim"] >= 2), v["arg2_ndim"] >= 2)))
)

def rule_149_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 149
        rule_149(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_149(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
