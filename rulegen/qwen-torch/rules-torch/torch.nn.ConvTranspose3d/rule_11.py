import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor shape should be valid for ConvTranspose3d (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(And(And(And(And(Select(v["arg1_shape"], i) > 0, 1 <= v["arg2_value"]), v["arg2_value"] <= 5), 1 <= v["arg3_value"]), v["arg3_value"] <= 5), 1 <= v["arg4_value"]), v["arg4_value"] <= 5), 1 <= v["arg5_value"]), v["arg5_value"] <= 5)) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(And(And(And(And(Select(v["arg1_shape"], i) > 0, 1 <= v["arg2_value"]), v["arg2_value"] <= 5), 1 <= v["arg3_value"]), v["arg3_value"] <= 5), 1 <= v["arg4_value"]), v["arg4_value"] <= 5), 1 <= v["arg5_value"]), v["arg5_value"] <= 5)) for i in range(6)]))
)

def rule_11_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))

        # Constraints for rule 11
        rule_11(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
