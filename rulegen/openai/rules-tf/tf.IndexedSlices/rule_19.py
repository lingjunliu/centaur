import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tuple dense_shape matches rank and dimensions; indices in range (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_ndim"] >= 1, 0 <= Select(v["arg2_range"], 0)), v["arg3_length"] == v["arg1_ndim"]), Select(v["arg3_values"], 0) > Select(v["arg2_range"], 1)), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) == Select(v["arg1_shape"], i)) for i in range(6)]))) if n else
          And(And(And(And(v["arg1_ndim"] >= 1, 0 <= Select(v["arg2_range"], 0)), v["arg3_length"] == v["arg1_ndim"]), Select(v["arg3_values"], 0) > Select(v["arg2_range"], 1)), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) == Select(v["arg1_shape"], i)) for i in range(6)])))
)

def rule_19_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 19
        rule_19(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
