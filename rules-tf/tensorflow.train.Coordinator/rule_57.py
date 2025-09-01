import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Parameter `clean_stop_exception_types` cannot contain dtype (index 12 (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(And((And([Implies(v_2 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_2) != 11) for v_2 in range(6)])), (And([Implies(v_2 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_2) != 12) for v_2 in range(6)])))) if n else
          And((And([Implies(v_2 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_2) != 11) for v_2 in range(6)])), (And([Implies(v_2 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_2) != 12) for v_2 in range(6)]))))
)

def rule_57_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 57
        rule_57(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)
