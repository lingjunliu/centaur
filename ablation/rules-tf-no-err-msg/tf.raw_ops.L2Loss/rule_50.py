import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the name is not none, then the number of elements in the tensor should be a power of 2 (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 6, Or([And(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) > 0, (And(Select(v["arg2_shape"], i), (Select(v["arg2_shape"], i) - 1))) == 0)) for i in range(6)]), True)) if n else
          If(v["arg1_value"] != 6, Or([And(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) > 0, (And(Select(v["arg2_shape"], i), (Select(v["arg2_shape"], i) - 1))) == 0)) for i in range(6)]), True))
)

def rule_50_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
