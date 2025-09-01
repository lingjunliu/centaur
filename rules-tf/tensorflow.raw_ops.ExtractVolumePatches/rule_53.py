import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# all elements in strides should be greater than 0 except strides[0] and strides[4], where they must be equal to 1 (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (3 + 1), And(And(Select(v["arg1_values"], i) > 0, Select(v["arg1_values"], 0) == 1), Select(v["arg1_values"], 4) == 1)) for i in range(6)])) if n else
          And([Implies(i < (3 + 1), And(And(Select(v["arg1_values"], i) > 0, Select(v["arg1_values"], 0) == 1), Select(v["arg1_values"], 4) == 1)) for i in range(6)]))
)

def rule_53_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 53
        rule_53(solver, {'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_values': arg1['values']}, neg)
