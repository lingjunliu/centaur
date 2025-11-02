import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Elements in dim list must be unique, non-negative and less than ndim(input (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] != 0, And((And([Implies(v_1 < (v["arg1_length"] - 1 + 1), (And([Implies(v_2 < (v["arg1_length"] - 1 + 1), If(v_1 != v_2, Select(v["arg1_values"], v_1) != Select(v["arg1_values"], v_2), True)) for v_2 in range(6)]))) for v_1 in range(6)])), (And([Implies(v_3 < (v["arg1_length"] - 1 + 1), (And(Select(v["arg1_values"], v_3) >= 0, Select(v["arg1_values"], v_3) < v["arg2_ndim"]))) for v_3 in range(6)]))), True)) if n else
          If(v["arg1_length"] != 0, And((And([Implies(v_1 < (v["arg1_length"] - 1 + 1), (And([Implies(v_2 < (v["arg1_length"] - 1 + 1), If(v_1 != v_2, Select(v["arg1_values"], v_1) != Select(v["arg1_values"], v_2), True)) for v_2 in range(6)]))) for v_1 in range(6)])), (And([Implies(v_3 < (v["arg1_length"] - 1 + 1), (And(Select(v["arg1_values"], v_3) >= 0, Select(v["arg1_values"], v_3) < v["arg2_ndim"]))) for v_3 in range(6)]))), True))
)

def rule_100_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
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

        # Constraints for rule 100
        rule_100(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim']}, neg)
