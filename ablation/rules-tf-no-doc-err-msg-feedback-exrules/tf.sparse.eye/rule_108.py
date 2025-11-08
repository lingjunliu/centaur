import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The product of the dimensions in batch_shape should not be excessively large (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, And((And([Implies(v_2 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_2) < 2048) for v_2 in range(6)])), (And([Implies(v_3 < (v["arg1_length"] - 1 + 1), (Or([And(v_4 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_3) * Select(v["arg1_values"], v_4) < 1000000) for v_4 in range(6)]))) for v_3 in range(6)]))), True)) if n else
          If(v["arg1_length"] > 0, And((And([Implies(v_2 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_2) < 2048) for v_2 in range(6)])), (And([Implies(v_3 < (v["arg1_length"] - 1 + 1), (Or([And(v_4 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_3) * Select(v["arg1_values"], v_4) < 1000000) for v_4 in range(6)]))) for v_3 in range(6)]))), True))
)

def rule_108_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 108
        rule_108(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
