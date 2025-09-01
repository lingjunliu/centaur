import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If beta is not near 0.5 then alpha has to be smaller than 20, for normalization to have an effect. bias must be reasonable. Dtype should also not be bfloat 16, and input should not be empty. (Rule 131)

rule_131 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] < 0.4, v["arg2_value"] > 0.6), And(And(And(v["arg1_value"] < 20, v["arg3_value"] < 1000), v["arg4_dtype"] != 13), And([Implies(i < (v["arg4_ndim"] - 1 + 1), Select(v["arg4_shape"], i) > 0) for i in range(6)])), True)) if n else
          If(Or(v["arg2_value"] < 0.4, v["arg2_value"] > 0.6), And(And(And(v["arg1_value"] < 20, v["arg3_value"] < 1000), v["arg4_dtype"] != 13), And([Implies(i < (v["arg4_ndim"] - 1 + 1), Select(v["arg4_shape"], i) > 0) for i in range(6)])), True))
)

def rule_131_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 131
        rule_131(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_131(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim'], 'arg4_dtype': arg4['dtype']}, neg)
