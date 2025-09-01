import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If large positive numbers are exponentiated, overflow problems are more likely (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), Select(v["arg1_range"], 1) < 5, If(v["arg1_dtype"] == 6, Select(v["arg1_range"], 1) < 11, If(v["arg1_dtype"] == 7, Select(v["arg1_range"], 1) < 80, If(v["arg1_dtype"] == 8, Select(v["arg1_range"], 1) < 700, True))))) if n else
          If(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), Select(v["arg1_range"], 1) < 5, If(v["arg1_dtype"] == 6, Select(v["arg1_range"], 1) < 11, If(v["arg1_dtype"] == 7, Select(v["arg1_range"], 1) < 80, If(v["arg1_dtype"] == 8, Select(v["arg1_range"], 1) < 700, True)))))
)

def rule_37_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 37
        rule_37(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']}, neg)
