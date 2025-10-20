import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The `value` tensor must have a numeric or boolean dtype. If it has at least one dimension, each dimension must be greater than 0 and not equal to -1. Dtype constraints are irrelevant if the tensor is empty or has no dimensions. (Rule 114)

rule_114 = lambda s, v, n=False: (
    s.add(Not((If(v["arg1_ndim"] > 0, And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) > 0, Select(v["arg1_shape"], i) != -1)) for i in range(6)])), (And(0 <= v["arg1_dtype"], v["arg1_dtype"] <= 11))), If(v["arg1_ndim"] == 0, (And(0 <= v["arg1_dtype"], v["arg1_dtype"] <= 11)), True)))) if n else
          (If(v["arg1_ndim"] > 0, And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) > 0, Select(v["arg1_shape"], i) != -1)) for i in range(6)])), (And(0 <= v["arg1_dtype"], v["arg1_dtype"] <= 11))), If(v["arg1_ndim"] == 0, (And(0 <= v["arg1_dtype"], v["arg1_dtype"] <= 11)), True))))
)

def rule_114_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 114
        rule_114(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_114(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']}, neg)
