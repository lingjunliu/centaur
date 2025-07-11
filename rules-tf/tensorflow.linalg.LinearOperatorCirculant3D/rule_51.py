import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the spectrum is Hermitian then the values for n0, n1, n2 have to be between negative and positive N0, N1, N2 (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((-1) * Select(v["arg4_shape"], v["arg4_ndim"] - 3) <= v["arg1_value"], v["arg1_value"] <= Select(v["arg4_shape"], v["arg4_ndim"] - 3)), (-1) * Select(v["arg4_shape"], v["arg4_ndim"] - 2) <= v["arg2_value"]), v["arg2_value"] <= Select(v["arg4_shape"], v["arg4_ndim"] - 2)), (-1) * Select(v["arg4_shape"], v["arg4_ndim"] - 1) <= v["arg3_value"]), v["arg3_value"] <= Select(v["arg4_shape"], v["arg4_ndim"] - 1))) if n else
          And(And(And(And(And((-1) * Select(v["arg4_shape"], v["arg4_ndim"] - 3) <= v["arg1_value"], v["arg1_value"] <= Select(v["arg4_shape"], v["arg4_ndim"] - 3)), (-1) * Select(v["arg4_shape"], v["arg4_ndim"] - 2) <= v["arg2_value"]), v["arg2_value"] <= Select(v["arg4_shape"], v["arg4_ndim"] - 2)), (-1) * Select(v["arg4_shape"], v["arg4_ndim"] - 1) <= v["arg3_value"]), v["arg3_value"] <= Select(v["arg4_shape"], v["arg4_ndim"] - 1)))
)

def rule_51_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 51
        rule_51(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape']}, neg)
