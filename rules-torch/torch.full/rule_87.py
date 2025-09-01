import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if `out` is specified, its shape must be compatible with `size` and have the correct dtype (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == v["arg2_length"], And([Implies(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_values"], i), If(v["arg3_value"] == 12, True, v["arg1_dtype"] == v["arg3_value"]))) for i in range(6)]))) if n else
          And(v["arg1_ndim"] == v["arg2_length"], And([Implies(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_values"], i), If(v["arg3_value"] == 12, True, v["arg1_dtype"] == v["arg3_value"]))) for i in range(6)])))
)

def rule_87_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 87
        rule_87(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value']}, neg)
