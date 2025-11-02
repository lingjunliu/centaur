import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If specified, the perm argument to sparse_transpose must contain a permutation of [0, 1, ..., rank - 1] (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, And([Implies(v_3 < (v["arg1_length"] - 1 + 1), Or([And(v_4 < (v["arg2_ndim"] - 1 + 1), Select(v["arg1_values"], v_3) == v_4) for v_4 in range(6)])) for v_3 in range(6)]), True)) if n else
          If(v["arg1_length"] > 0, And([Implies(v_3 < (v["arg1_length"] - 1 + 1), Or([And(v_4 < (v["arg2_ndim"] - 1 + 1), Select(v["arg1_values"], v_3) == v_4) for v_4 in range(6)])) for v_3 in range(6)]), True))
)

def rule_68_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 68
        rule_68(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim']}, neg)
