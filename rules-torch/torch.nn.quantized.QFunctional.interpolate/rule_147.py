import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When using area mode then the input and output shapes cannot be the same when the recompute scale factor is set to false (Rule 147)

rule_147 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 29, v["arg4_value"] == False), And(v["arg2_ndim"] == v["arg3_length"], Or([And(i < (v["arg3_length"] - 1 + 1), Select(v["arg2_shape"], i) != Select(v["arg3_values"], i)) for i in range(6)])), True)) if n else
          If(And(v["arg1_value"] == 29, v["arg4_value"] == False), And(v["arg2_ndim"] == v["arg3_length"], Or([And(i < (v["arg3_length"] - 1 + 1), Select(v["arg2_shape"], i) != Select(v["arg3_values"], i)) for i in range(6)])), True))
)

def rule_147_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == arg4)

        # Constraints for rule 147
        rule_147(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_147(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value']}, neg)
