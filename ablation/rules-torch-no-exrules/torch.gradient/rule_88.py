import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dim is specified, elements are unique and non-negative. Also, each element should be less than ndim(input (Rule 88)

rule_88 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] != 0, And((And([Implies(v_1 < (v["arg2_length"] - 1 + 1), (And([Implies(v_2 < (v["arg2_length"] - 1 + 1), If(v_1 != v_2, Select(v["arg2_values"], v_1) != Select(v["arg2_values"], v_2), True)) for v_2 in range(6)]))) for v_1 in range(6)])), (And([Implies(v_3 < (v["arg2_length"] - 1 + 1), (And(Select(v["arg2_values"], v_3) >= 0, Select(v["arg2_values"], v_3) < v["arg1_ndim"]))) for v_3 in range(6)]))), True)) if n else
          If(v["arg2_length"] != 0, And((And([Implies(v_1 < (v["arg2_length"] - 1 + 1), (And([Implies(v_2 < (v["arg2_length"] - 1 + 1), If(v_1 != v_2, Select(v["arg2_values"], v_1) != Select(v["arg2_values"], v_2), True)) for v_2 in range(6)]))) for v_1 in range(6)])), (And([Implies(v_3 < (v["arg2_length"] - 1 + 1), (And(Select(v["arg2_values"], v_3) >= 0, Select(v["arg2_values"], v_3) < v["arg1_ndim"]))) for v_3 in range(6)]))), True))
)

def rule_88_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 88
        rule_88(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_88(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
