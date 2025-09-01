import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# To work, its numerical and not a boolean or complex  (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or((v["arg1_dtype"] == 0), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10)), False, Or([And(i < (If(0 < 0, 0, 0) + 1), Select(v["arg1_shape"], i) >= 0) for i in range(6)]))) if n else
          If(Or(Or((v["arg1_dtype"] == 0), (v["arg1_dtype"] == 9)), (v["arg1_dtype"] == 10)), False, Or([And(i < (If(0 < 0, 0, 0) + 1), Select(v["arg1_shape"], i) >= 0) for i in range(6)])))
)

def rule_115_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 115
        rule_115(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
