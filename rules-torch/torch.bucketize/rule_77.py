import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If an out tensor is passed in and is numerical , and if there is >1 element in boundaries, out_int32 conformance required (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg2_ndim"] > 0, (And(v["arg2_dtype"] > 0, v["arg2_dtype"] < 9))), Select(v["arg3_shape"], 0) > 1), (If(v["arg1_value"] == True, v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), False)) if n else
          If(And(And(v["arg2_ndim"] > 0, (And(v["arg2_dtype"] > 0, v["arg2_dtype"] < 9))), Select(v["arg3_shape"], 0) > 1), (If(v["arg1_value"] == True, v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), False))
)

def rule_77_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 77
        rule_77(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_shape': arg3['shape']}, neg)
