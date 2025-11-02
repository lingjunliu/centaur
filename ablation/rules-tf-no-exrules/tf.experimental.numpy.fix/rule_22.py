import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dtype is complex, ensure that the absolute value of the elements is within a certain range, simplified bound (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), And([Implies(v_1 < (100 + 1), (v_1 * v_1) < 1000000) for v_1 in range(6)]), True)) if n else
          If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), And([Implies(v_1 < (100 + 1), (v_1 * v_1) < 1000000) for v_1 in range(6)]), True))
)

def rule_22_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 22
        rule_22(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_dtype': arg1['dtype']}, neg)
