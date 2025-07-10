import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If track running stats is not enabled and tensor is really large and a huge number of features , you should disable affine  (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] == False, v["arg2_value"] > 2048), Select(v["arg4_shape"], 0) * Select(v["arg4_shape"], 1) * Select(v["arg4_shape"], 2) * Select(v["arg4_shape"], 3) * Select(v["arg4_shape"], 4) > 100000), v["arg3_value"] == False, False)) if n else
          If(And(And(v["arg1_value"] == False, v["arg2_value"] > 2048), Select(v["arg4_shape"], 0) * Select(v["arg4_shape"], 1) * Select(v["arg4_shape"], 2) * Select(v["arg4_shape"], 3) * Select(v["arg4_shape"], 4) > 100000), v["arg3_value"] == False, False))
)

def rule_103_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == arg3)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 103
        rule_103(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape']}, neg)
