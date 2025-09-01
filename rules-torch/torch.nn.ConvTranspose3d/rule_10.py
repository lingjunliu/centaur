import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size, stride, padding, output_padding, dilation must have same length if they are tuples (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(And(And(And((v["arg1_length"] == v["arg2_length"]), (v["arg1_length"] == v["arg3_length"])), (v["arg1_length"] == v["arg4_length"])), (v["arg1_length"] == v["arg5_length"]))) if n else
          And(And(And((v["arg1_length"] == v["arg2_length"]), (v["arg1_length"] == v["arg3_length"])), (v["arg1_length"] == v["arg4_length"])), (v["arg1_length"] == v["arg5_length"])))
)

def rule_10_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
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
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')
        arg4_length = Int('arg4_length')
        arg5_length = Int('arg5_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_length == len(arg4))
        solver.add(arg5_length == len(arg5))

        # Constraints for rule 10
        rule_10(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_length': arg3_length, 'arg4_length': arg4_length, 'arg5_length': arg5_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length'], 'arg4_length': arg4['length'], 'arg5_length': arg5['length']}, neg)
