import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# handle zero output size for tuple inputs (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg2_length"] == 2, v["arg3_length"] == 2), v["arg4_length"] == 2), (If(v["arg5_value"] == False, (And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) > Select(v["arg2_values"], 0) - 1 + 2 * Select(v["arg3_values"], 0) / Select(v["arg4_values"], 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > Select(v["arg2_values"], 1) - 1 + 2 * Select(v["arg3_values"], 1) / Select(v["arg4_values"], 1))), True)))) if n else
          And(And(And(v["arg2_length"] == 2, v["arg3_length"] == 2), v["arg4_length"] == 2), (If(v["arg5_value"] == False, (And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) > Select(v["arg2_values"], 0) - 1 + 2 * Select(v["arg3_values"], 0) / Select(v["arg4_values"], 0), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > Select(v["arg2_values"], 1) - 1 + 2 * Select(v["arg3_values"], 1) / Select(v["arg4_values"], 1))), True))))
)

def rule_12_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_value == arg5)

        # Constraints for rule 12
        rule_12(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_length': arg4_length, 'arg4_values': arg4_values, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_length': arg4['length'], 'arg4_values': arg4['values'], 'arg5_value': arg5['value']}, neg)
