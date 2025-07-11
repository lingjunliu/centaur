import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the output dtype is string, and input is also string, the length of the input should be small. (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg2_value"] != 11), (If((v["arg1_dtype"] == 11), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], i) - 1 + 1), Select(v["arg1_range"], 1) < 255) for j in range(6)])) for i in range(6)]), False)))) if n else
          Or((v["arg2_value"] != 11), (If((v["arg1_dtype"] == 11), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], i) - 1 + 1), Select(v["arg1_range"], 1) < 255) for j in range(6)])) for i in range(6)]), False))))
)

def rule_51_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 51
        rule_51(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
