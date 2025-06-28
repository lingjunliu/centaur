import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if float v_1 is smaller than the min value and larger than -1 for tensor v_2, then result of multiplication of min value of v_2 with -1 has to smaller than 100 (Rule 619)

rule_619 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] < Select(v["arg2_range"], 0), Select(v["arg2_range"], 0) > 0), v["arg2_ndim"] > 0), 0 - Select(v["arg2_range"], 0) < 100, False)) if n else
          If(And(And(v["arg1_value"] < Select(v["arg2_range"], 0), Select(v["arg2_range"], 0) > 0), v["arg2_ndim"] > 0), 0 - Select(v["arg2_range"], 0) < 100, False))
)

def rule_619_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 619
        rule_619(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_619(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_range': arg2['range']}, neg)
