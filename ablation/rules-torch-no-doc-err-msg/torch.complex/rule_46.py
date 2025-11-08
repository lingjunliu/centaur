import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is given, and both real and imag are also given, it must match the real and imag types (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] > 0, v["arg3_ndim"] > 0), (If(Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), v["arg1_value"] == v["arg2_dtype"], If(Or(v["arg3_dtype"] == 9, v["arg3_dtype"] == 10), v["arg1_value"] == v["arg3_dtype"], True))), True)) if n else
          If(And(v["arg2_ndim"] > 0, v["arg3_ndim"] > 0), (If(Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10), v["arg1_value"] == v["arg2_dtype"], If(Or(v["arg3_dtype"] == 9, v["arg3_dtype"] == 10), v["arg1_value"] == v["arg3_dtype"], True))), True))
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)
