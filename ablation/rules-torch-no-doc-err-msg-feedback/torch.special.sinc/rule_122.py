import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if dimension value is in range, list contains numbers to be great or equal minimum value and smaller or equal to maximum values of the tensor. (Rule 122)

rule_122 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg2_value"] > 0, v["arg2_value"] < v["arg1_ndim"])), And([Implies(i < (v["arg3_length"] - 1 + 1), And(Select(v["arg1_range"], 0) <= Select(v["arg3_values"], i), Select(v["arg3_values"], i) <= Select(v["arg1_range"], 1))) for i in range(6)]), True)) if n else
          If((And(v["arg2_value"] > 0, v["arg2_value"] < v["arg1_ndim"])), And([Implies(i < (v["arg3_length"] - 1 + 1), And(Select(v["arg1_range"], 0) <= Select(v["arg3_values"], i), Select(v["arg3_values"], i) <= Select(v["arg1_range"], 1))) for i in range(6)]), True))
)

def rule_122_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, list) and all(isinstance(e, (float, np.floating)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 122
        rule_122(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_122(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
