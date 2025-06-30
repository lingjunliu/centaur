import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If int v_1 is between -10 and 10, and string v_2 is not 'none' and tensor v_3 has shape[0] equal to 10 then the result of multiplication between integer v_1 and float 0.5 must be smaller or equal to max of tensor v_3 (Rule 588)

rule_588 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_value"] > -10, v["arg1_value"] < 10), v["arg2_value"] != 6), Select(v["arg3_shape"], 0) == 10), v["arg1_value"] * 0.5 <= Select(v["arg3_range"], 1), False)) if n else
          If(And(And(And(v["arg1_value"] > -10, v["arg1_value"] < 10), v["arg2_value"] != 6), Select(v["arg3_shape"], 0) == 10), v["arg1_value"] * 0.5 <= Select(v["arg3_range"], 1), False))
)

def rule_588_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values.index(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 588
        rule_588(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_588(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_range': arg3['range']}, neg)
