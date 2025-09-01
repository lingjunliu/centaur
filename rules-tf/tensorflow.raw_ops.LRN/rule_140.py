import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If image is smaller than 16x16 then alpha should be less than 2 and beta smaller than 0.5 and depth radius smaller than 3, AND channels >2. Data type needs to be consistent (FP32 for small images (Rule 140)

rule_140 = lambda s, v, n=False: (
    s.add(Not(If(And(And(Select(v["arg1_shape"], 2) < 16, Select(v["arg1_shape"], 3) < 16), Select(v["arg1_shape"], 3) > 2), And(And(And(And(v["arg2_value"] < 2, v["arg3_value"] < 0.5), v["arg4_value"] < 3), v["arg1_dtype"] == 7), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) > 0, v["arg2_value"] > 0.0001)) for i in range(6)])), True)) if n else
          If(And(And(Select(v["arg1_shape"], 2) < 16, Select(v["arg1_shape"], 3) < 16), Select(v["arg1_shape"], 3) > 2), And(And(And(And(v["arg2_value"] < 2, v["arg3_value"] < 0.5), v["arg4_value"] < 3), v["arg1_dtype"] == 7), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) > 0, v["arg2_value"] > 0.0001)) for i in range(6)])), True))
)

def rule_140_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 140
        rule_140(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_140(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
