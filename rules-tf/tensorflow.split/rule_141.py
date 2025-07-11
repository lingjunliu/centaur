import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When num_or_size_splits is a list the sum of its element must be equal to the shape of value at axis (Rule 141)

rule_141 = lambda s, v, n=False: (
    s.add(Not(And(And((0 <= v["arg3_value"]), (v["arg3_value"] < v["arg1_ndim"])), (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 0) for i in range(6)])))) if n else
          And(And((0 <= v["arg3_value"]), (v["arg3_value"] < v["arg1_ndim"])), (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 0) for i in range(6)]))))
)

def rule_141_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 141
        rule_141(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_141(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
