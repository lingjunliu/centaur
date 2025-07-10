import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If A does not have the appropriate dtype, error will be thrown. Reformulation with quantifiers and comparisons (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(And([Implies(dt < (10 + 1), If(v["arg1_dtype"] == dt, True, Or(Or(Or(Or(False, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))) for dt in range(6)])) if n else
          And([Implies(dt < (10 + 1), If(v["arg1_dtype"] == dt, True, Or(Or(Or(Or(False, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))) for dt in range(6)]))
)

def rule_46_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype']}, neg)
