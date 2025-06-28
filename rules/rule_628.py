import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If string v_1 is 'tanh' and integer v_2 = 1, then minimum element of tensor v_3 has to be greater than -10 and less than 1 (Rule 628)

rule_628 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] == 11, v["arg2_value"] == 1), v["arg3_ndim"] > 0), And(Select(v["arg3_range"], 0) > -10, Select(v["arg3_range"], 0) < 1), False)) if n else
          If(And(And(v["arg1_value"] == 11, v["arg2_value"] == 1), v["arg3_ndim"] > 0), And(Select(v["arg3_range"], 0) > -10, Select(v["arg3_range"], 0) < 1), False))
)

def rule_628_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 628
        rule_628(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_628(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim']}, neg)
