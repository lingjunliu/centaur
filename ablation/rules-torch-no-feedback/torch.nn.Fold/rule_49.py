import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check if dimension of output_size, kernel_size, dilation, padding and stride are 1 if input dimension is 3 (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 3, And(And(And((v["arg2_length"] == 1), (v["arg3_length"] == 1)), (v["arg4_length"] == 1)), (v["arg5_length"] == 1)), True)) if n else
          If(v["arg1_ndim"] == 3, And(And(And((v["arg2_length"] == 1), (v["arg3_length"] == 1)), (v["arg4_length"] == 1)), (v["arg5_length"] == 1)), True))
)

def rule_49_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')
        arg4_length = Int('arg4_length')
        arg5_length = Int('arg5_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_length == len(arg4))
        solver.add(arg5_length == len(arg5))

        # Constraints for rule 49
        rule_49(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_length': arg3_length, 'arg4_length': arg4_length, 'arg5_length': arg5_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length'], 'arg4_length': arg4['length'], 'arg5_length': arg5['length']}, neg)
