import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Images must be a 4D tensor of the allowed dtypes, size must be 1D and of int32, size elements have to be non-negative, align_corners is boolean and if its true then the height and width of images must be >= 2 and the elements of size must be >= 2 (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 4, (Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 12))), v["arg2_ndim"] == 1), v["arg2_dtype"] == 3), Select(v["arg2_shape"], 0) == 2), Select(v["arg2_shape"], 1) >= 0), Select(v["arg2_shape"], 0) >= 0), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), If(v["arg3_value"] == True, And(And(And(Select(v["arg1_shape"], 1) >= 2, Select(v["arg1_shape"], 2) >= 2), Select(v["arg2_shape"], 0) >= 2), Select(v["arg2_shape"], 1) >= 2), True))) if n else
          And(And(And(And(And(And(And(And(v["arg1_ndim"] == 4, (Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 12))), v["arg2_ndim"] == 1), v["arg2_dtype"] == 3), Select(v["arg2_shape"], 0) == 2), Select(v["arg2_shape"], 1) >= 0), Select(v["arg2_shape"], 0) >= 0), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), If(v["arg3_value"] == True, And(And(And(Select(v["arg1_shape"], 1) >= 2, Select(v["arg1_shape"], 2) >= 2), Select(v["arg2_shape"], 0) >= 2), Select(v["arg2_shape"], 1) >= 2), True)))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
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

        # Constraints for rule 48
        rule_48(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
