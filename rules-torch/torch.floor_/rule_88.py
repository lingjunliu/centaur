import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input tensor has dimensions and they are all +ve , and excluding the conditions where it cannot be a boolean or a complex tensor , for floor calculation to be validated. (Rule 88)

rule_88 = lambda s, v, n=False: (
    s.add(Not(And((Or([And(i < (0 + 1), (And([Implies(j < (0 + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)]))) for i in range(6)])), (And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 9))))) if n else
          And((Or([And(i < (0 + 1), (And([Implies(j < (0 + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)]))) for i in range(6)])), (And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 9)))))
)

def rule_88_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 88
        rule_88(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_88(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
