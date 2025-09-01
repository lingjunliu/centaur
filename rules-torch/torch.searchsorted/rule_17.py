import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Output tensor's dtype must be int32 if out_int32 is True, int64 otherwise - checking output shape as well (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(And((If(v["arg1_value"], v["arg2_dtype"] == 2, v["arg2_dtype"] == 4)), (And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i)) for i in range(6)])))) if n else
          And((If(v["arg1_value"], v["arg2_dtype"] == 2, v["arg2_dtype"] == 4)), (And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i)) for i in range(6)]))))
)

def rule_17_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 17
        rule_17(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)
