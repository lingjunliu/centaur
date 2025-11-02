import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tensor is all NaNs, and dtype is specified then the output dtype must be compatible with Nan. (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If((And([Implies(i < (v["arg1_ndim"] - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], i) - 1 + 1), True) for j in range(6)])) for i in range(6)])), (Or(Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 12)), True)) if n else
          If((And([Implies(i < (v["arg1_ndim"] - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], i) - 1 + 1), True) for j in range(6)])) for i in range(6)])), (Or(Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 12)), True))
)

def rule_65_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 65
        rule_65(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
