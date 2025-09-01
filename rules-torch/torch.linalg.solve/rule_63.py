import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# A and B must have same dtype, must be floating or complex, A needs at least 2 dimensions and be square, and compatible shapes based on left (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(v["arg1_dtype"] == v["arg2_dtype"], (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 11))), v["arg1_ndim"] >= 2), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == Select(v["arg1_shape"], v["arg1_ndim"] - 2)), (If(And(And(v["arg3_value"] == True, v["arg1_ndim"] == 2), v["arg2_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), True))), (If(And(And(v["arg3_value"] == True, v["arg1_ndim"] == 2), v["arg2_ndim"] == 2), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), True))), (If(And(And(v["arg3_value"] == False, v["arg1_ndim"] == 2), v["arg2_ndim"] == 2), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1), True)))) if n else
          And(And(And(And(And(And(v["arg1_dtype"] == v["arg2_dtype"], (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 11))), v["arg1_ndim"] >= 2), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == Select(v["arg1_shape"], v["arg1_ndim"] - 2)), (If(And(And(v["arg3_value"] == True, v["arg1_ndim"] == 2), v["arg2_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), True))), (If(And(And(v["arg3_value"] == True, v["arg1_ndim"] == 2), v["arg2_ndim"] == 2), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), True))), (If(And(And(v["arg3_value"] == False, v["arg1_ndim"] == 2), v["arg2_ndim"] == 2), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1), True))))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)

        # Constraints for rule 63
        rule_63(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
