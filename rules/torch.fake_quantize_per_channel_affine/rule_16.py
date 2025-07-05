import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Input must be Float32, Scale must be Float32 1D tensor, Zero-point must be Int32/Float32/Half 1D tensor, quant_min < quant_max, axis within input dimension (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8), v["arg2_ndim"] == 1), (Or(Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8))), v["arg3_ndim"] == 1), v["arg5_value"] < v["arg6_value"]), 0 <= v["arg4_value"]), v["arg4_value"] < v["arg1_ndim"])) if n else
          And(And(And(And(And(And(And(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8), v["arg2_ndim"] == 1), (Or(Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 7), v["arg3_dtype"] == 8))), v["arg3_ndim"] == 1), v["arg5_value"] < v["arg6_value"]), 0 <= v["arg4_value"]), v["arg4_value"] < v["arg1_ndim"]))
)

def rule_16_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False
        if not (isinstance(arg6, (int, np.integer)) and not isinstance(arg6, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))

        # Constraints for rule 16
        rule_16(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
