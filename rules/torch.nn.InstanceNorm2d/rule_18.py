import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If using running_stats, the input tensor dtype and running_var dtype must match; and using affine, gamma and beta dtype must match with input tensor dtype (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], And((v["arg2_dtype"] == v["arg3_dtype"]), If(v["arg4_value"], (And(v["arg5_dtype"] == v["arg2_dtype"], v["arg6_dtype"] == v["arg2_dtype"])), False)), False)) if n else
          If(v["arg1_value"], And((v["arg2_dtype"] == v["arg3_dtype"]), If(v["arg4_value"], (And(v["arg5_dtype"] == v["arg2_dtype"], v["arg6_dtype"] == v["arg2_dtype"])), False)), False))
)

def rule_18_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Bool('arg4_value')
        arg5_dtype = Int('arg5_dtype')
        arg6_dtype = Int('arg6_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == arg4)
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))
        solver.add(arg6_dtype == list_of_available_dtypes.index(arg6.dtype))

        # Constraints for rule 18
        rule_18(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value, 'arg5_dtype': arg5_dtype, 'arg6_dtype': arg6_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value'], 'arg5_dtype': arg5['dtype'], 'arg6_dtype': arg6['dtype']}, neg)
