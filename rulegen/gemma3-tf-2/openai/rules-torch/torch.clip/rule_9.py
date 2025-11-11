import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# mixed bounds: float min and tensor max are consistent and broadcastable (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg3_dtype"] == v["arg1_dtype"], (And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10))), v["arg3_ndim"] == v["arg1_ndim"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Or(Select(v["arg3_shape"], i) == 1, Select(v["arg3_shape"], i) == Select(v["arg1_shape"], i)), v["arg2_value"] <= Select(v["arg3_range"], 0))) for i in range(6)]))) if n else
          And(And(And(v["arg3_dtype"] == v["arg1_dtype"], (And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10))), v["arg3_ndim"] == v["arg1_ndim"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Or(Select(v["arg3_shape"], i) == 1, Select(v["arg3_shape"], i) == Select(v["arg1_shape"], i)), v["arg2_value"] <= Select(v["arg3_range"], 0))) for i in range(6)])))
)

def rule_9_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 9
        rule_9(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg3_range': arg3['range']}, neg)
