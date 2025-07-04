import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If input is a str type, then its values should be selected from the defined string list. (Rule 7)

rule_7 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 0) == 1), Select(v["arg1_range"], 0) == 2), Select(v["arg1_range"], 0) == 3), Select(v["arg1_range"], 0) == 4), Select(v["arg1_range"], 0) == 5), Select(v["arg1_range"], 0) == 6), Select(v["arg1_range"], 0) == 7), Select(v["arg1_range"], 0) == 8), Select(v["arg1_range"], 0) == 9), Select(v["arg1_range"], 0) == 10), Select(v["arg1_range"], 0) == 11)), False)) if n else
          If(v["arg1_dtype"] == 11, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 0) == 1), Select(v["arg1_range"], 0) == 2), Select(v["arg1_range"], 0) == 3), Select(v["arg1_range"], 0) == 4), Select(v["arg1_range"], 0) == 5), Select(v["arg1_range"], 0) == 6), Select(v["arg1_range"], 0) == 7), Select(v["arg1_range"], 0) == 8), Select(v["arg1_range"], 0) == 9), Select(v["arg1_range"], 0) == 10), Select(v["arg1_range"], 0) == 11)), False))
)

def rule_7_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 7
        rule_7(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_7(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
