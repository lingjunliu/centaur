import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If shift is a tuple/list, axes must also be a tuple/list of the same length and less than ndim (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] > 0, And(v["arg2_length"] == v["arg3_length"], And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) < v["arg1_ndim"]) for i in range(6)])), False)) if n else
          If(v["arg2_length"] > 0, And(v["arg2_length"] == v["arg3_length"], And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) < v["arg1_ndim"]) for i in range(6)])), False))
)

def rule_14_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 14
        rule_14(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
