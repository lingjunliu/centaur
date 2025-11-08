import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If num_rows, num_cols, and batch_shape are small and validate_indices is true, then dtype should not be complex128 for performance reasons (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg4_value"] == True, v["arg1_value"] < 100), v["arg2_value"] < 100), v["arg3_length"] < 2), v["arg5_value"] != 10, True)) if n else
          If(And(And(And(v["arg4_value"] == True, v["arg1_value"] < 100), v["arg2_value"] < 100), v["arg3_length"] < 2), v["arg5_value"] != 10, True))
)

def rule_74_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, bool):
            return False
        if not (isinstance(arg5, torch.dtype) or isinstance(arg5, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg4_value = Bool('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == list_of_available_dtypes.index(np_dtype(arg5)))

        # Constraints for rule 74
        rule_74(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
