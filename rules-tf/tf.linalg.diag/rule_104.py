import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The number of diagonals in diagonal must equal d_upper - d_lower + 1 (diag rule)
# {v_1 : tensor, v_2 : tensor} |= shape(v_1, -2) = v_2[1] - v_2[0] + 1

rule_diag_k = lambda s, v, n=False: (
    s.add(Not(
        Select(v["arg1_shape"], v["arg1_ndim"] - 2) == Select(v["arg2_value"], 1) - Select(v["arg2_value"], 0) + 1
    ) if n else
        Select(v["arg1_shape"], v["arg1_ndim"] - 2) == Select(v["arg2_value"], 1) - Select(v["arg2_value"], 0) + 1
    )
)

def rule_diag_k_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if arg2.shape[0] != 2:
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Array('arg2_value', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.shape[0]):
            arg2_value = Store(arg2_value, i, int(arg2[i]))

        # Constraints for diag k rule
        rule_diag_k(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_diag_k(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)