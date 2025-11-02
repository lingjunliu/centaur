import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If stride is None, assign the same value as kernel_size for each dimension (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(And([Implies(v_1 < (2 + 1), Select(v["arg2_values"], v_1) == Select(v["arg1_values"], v_1)) for v_1 in range(6)])) if n else
          And([Implies(v_1 < (2 + 1), Select(v["arg2_values"], v_1) == Select(v["arg1_values"], v_1)) for v_1 in range(6)]))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 26
        rule_26(solver, {'arg1_values': arg1_values, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_values': arg1['values'], 'arg2_values': arg2['values']}, neg)
