import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combined constraints, with reordered OR clauses (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(And(And(And((v["arg2_value"] >= 0), (v["arg2_value"] < 6)), (And(And(And((v["arg1_ndim"] == 2), (v["arg1_dtype"] == 11)), (Select(v["arg1_shape"], 1) == 3)), (Select(v["arg1_shape"], 0) > 0)))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((v["arg3_value"] == 12), (v["arg3_value"] == 0)), (v["arg3_value"] == 1)), (v["arg3_value"] == 2)), (v["arg3_value"] == 3)), (v["arg3_value"] == 4)), (v["arg3_value"] == 5)), (v["arg3_value"] == 6)), (v["arg3_value"] == 7)), (v["arg3_value"] == 8)), (v["arg3_value"] == 9)), (v["arg3_value"] == 10)), (v["arg3_value"] == 11))))) if n else
          And(And(And((v["arg2_value"] >= 0), (v["arg2_value"] < 6)), (And(And(And((v["arg1_ndim"] == 2), (v["arg1_dtype"] == 11)), (Select(v["arg1_shape"], 1) == 3)), (Select(v["arg1_shape"], 0) > 0)))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((v["arg3_value"] == 12), (v["arg3_value"] == 0)), (v["arg3_value"] == 1)), (v["arg3_value"] == 2)), (v["arg3_value"] == 3)), (v["arg3_value"] == 4)), (v["arg3_value"] == 5)), (v["arg3_value"] == 6)), (v["arg3_value"] == 7)), (v["arg3_value"] == 8)), (v["arg3_value"] == 9)), (v["arg3_value"] == 10)), (v["arg3_value"] == 11)))))
)

def rule_80_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 80
        rule_80(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
