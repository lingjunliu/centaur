import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When padding is VALID, then output spatial dimensions will be smaller or equal to input spatial dimensions, integer division (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] == 21, v["arg5_value"] == 24), And(And((Select(v["arg1_shape"], 1) - Select(v["arg2_values"], 1) + Select(v["arg3_values"], 1) - 1) / Select(v["arg3_values"], 1) <= Select(v["arg1_shape"], 1), (Select(v["arg1_shape"], 2) - Select(v["arg2_values"], 2) + Select(v["arg3_values"], 2) - 1) / Select(v["arg3_values"], 2) <= Select(v["arg1_shape"], 2)), (Select(v["arg1_shape"], 3) - Select(v["arg2_values"], 3) + Select(v["arg3_values"], 3) - 1) / Select(v["arg3_values"], 3) <= Select(v["arg1_shape"], 3)), If(And(v["arg4_value"] == 21, v["arg5_value"] == 25), And(And((Select(v["arg1_shape"], 2) - Select(v["arg2_values"], 1) + Select(v["arg3_values"], 1) - 1) / Select(v["arg3_values"], 1) <= Select(v["arg1_shape"], 2), (Select(v["arg1_shape"], 3) - Select(v["arg2_values"], 2) + Select(v["arg3_values"], 2) - 1) / Select(v["arg3_values"], 2) <= Select(v["arg1_shape"], 3)), (Select(v["arg1_shape"], 4) - Select(v["arg2_values"], 3) + Select(v["arg3_values"], 3) - 1) / Select(v["arg3_values"], 3) <= Select(v["arg1_shape"], 4)), True))) if n else
          If(And(v["arg4_value"] == 21, v["arg5_value"] == 24), And(And((Select(v["arg1_shape"], 1) - Select(v["arg2_values"], 1) + Select(v["arg3_values"], 1) - 1) / Select(v["arg3_values"], 1) <= Select(v["arg1_shape"], 1), (Select(v["arg1_shape"], 2) - Select(v["arg2_values"], 2) + Select(v["arg3_values"], 2) - 1) / Select(v["arg3_values"], 2) <= Select(v["arg1_shape"], 2)), (Select(v["arg1_shape"], 3) - Select(v["arg2_values"], 3) + Select(v["arg3_values"], 3) - 1) / Select(v["arg3_values"], 3) <= Select(v["arg1_shape"], 3)), If(And(v["arg4_value"] == 21, v["arg5_value"] == 25), And(And((Select(v["arg1_shape"], 2) - Select(v["arg2_values"], 1) + Select(v["arg3_values"], 1) - 1) / Select(v["arg3_values"], 1) <= Select(v["arg1_shape"], 2), (Select(v["arg1_shape"], 3) - Select(v["arg2_values"], 2) + Select(v["arg3_values"], 2) - 1) / Select(v["arg3_values"], 2) <= Select(v["arg1_shape"], 3)), (Select(v["arg1_shape"], 4) - Select(v["arg2_values"], 3) + Select(v["arg3_values"], 3) - 1) / Select(v["arg3_values"], 3) <= Select(v["arg1_shape"], 4)), True)))
)

def rule_63_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, str):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        solver.add(arg5_value == list_of_string_values_tf.index(arg5))

        # Constraints for rule 63
        rule_63(solver, {'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg3_values': arg3_values, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
