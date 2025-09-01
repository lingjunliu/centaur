import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# mat1 needs to store integer indices, be 2D, and have all shape parameters greater than 0, mat2 also has shape values > 0, all tensors must be a valid floating point dtype (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((Or(Or((v["arg1_dtype"] == 6), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8))), (Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)))), (Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5))), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0)), And([Implies(i < (1 + 1), And(Select(v["arg1_shape"], i) > 0, And([Implies(i < (1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))) for i in range(6)]))) if n else
          And(And(And(And((Or(Or((v["arg1_dtype"] == 6), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8))), (Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)))), (Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5))), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0)), And([Implies(i < (1 + 1), And(Select(v["arg1_shape"], i) > 0, And([Implies(i < (1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))) for i in range(6)])))
)

def rule_111_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 111
        rule_111(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
