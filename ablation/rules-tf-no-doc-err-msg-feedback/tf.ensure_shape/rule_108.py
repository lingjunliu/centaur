import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check the shape for at least one axis in the tensor v_1 against int v_2 using the provided comparision v_3 (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (v["arg1_ndim"] - 1 + 1), (If(v["arg3_value"] == 0, Select(v["arg1_shape"], i) == v["arg2_value"], If(v["arg3_value"] == 1, Select(v["arg1_shape"], i) != v["arg2_value"], If(v["arg3_value"] == 2, Select(v["arg1_shape"], i) > v["arg2_value"], If(v["arg3_value"] == 3, Select(v["arg1_shape"], i) < v["arg2_value"], If(v["arg3_value"] == 4, Select(v["arg1_shape"], i) >= v["arg2_value"], If(v["arg3_value"] == 5, Select(v["arg1_shape"], i) <= v["arg2_value"], False)))))))) for i in range(6)])) if n else
          Or([And(i < (v["arg1_ndim"] - 1 + 1), (If(v["arg3_value"] == 0, Select(v["arg1_shape"], i) == v["arg2_value"], If(v["arg3_value"] == 1, Select(v["arg1_shape"], i) != v["arg2_value"], If(v["arg3_value"] == 2, Select(v["arg1_shape"], i) > v["arg2_value"], If(v["arg3_value"] == 3, Select(v["arg1_shape"], i) < v["arg2_value"], If(v["arg3_value"] == 4, Select(v["arg1_shape"], i) >= v["arg2_value"], If(v["arg3_value"] == 5, Select(v["arg1_shape"], i) <= v["arg2_value"], False)))))))) for i in range(6)]))
)

def rule_108_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 108
        rule_108(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
