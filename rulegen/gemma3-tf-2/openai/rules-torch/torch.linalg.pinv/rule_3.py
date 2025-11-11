import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Valid float tolerances with out: shape/batch-compatible and nonnegative tolerances (Rule 3)

rule_3 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_ndim"] >= 2, v["arg2_value"] >= 0), v["arg3_value"] >= 0), v["arg5_ndim"] == v["arg1_ndim"]), Select(v["arg5_shape"], -2) == Select(v["arg1_shape"], -1)), Select(v["arg5_shape"], -1) == Select(v["arg1_shape"], -2)), (If(v["arg1_ndim"] >= 3, And([Implies(i < (v["arg1_ndim"] - 3 + 1), Select(v["arg5_shape"], i) == Select(v["arg1_shape"], i)) for i in range(6)]), True))), (Or(v["arg4_value"] == True, v["arg4_value"] == False)))) if n else
          And(And(And(And(And(And(And(v["arg1_ndim"] >= 2, v["arg2_value"] >= 0), v["arg3_value"] >= 0), v["arg5_ndim"] == v["arg1_ndim"]), Select(v["arg5_shape"], -2) == Select(v["arg1_shape"], -1)), Select(v["arg5_shape"], -1) == Select(v["arg1_shape"], -2)), (If(v["arg1_ndim"] >= 3, And([Implies(i < (v["arg1_ndim"] - 3 + 1), Select(v["arg5_shape"], i) == Select(v["arg1_shape"], i)) for i in range(6)]), True))), (Or(v["arg4_value"] == True, v["arg4_value"] == False))))
)

def rule_3_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_ndim = Int('arg5_ndim')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_ndim == arg5.ndim)
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 3
        rule_3(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_ndim': arg5_ndim, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_3(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_ndim': arg5['ndim'], 'arg5_shape': arg5['shape']}, neg)
