import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input tensor v_1's dtype must be float type if the input is not greater than or equal to -1 (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 0) < -1, (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), True)) if n else
          If(Select(v["arg1_range"], 0) < -1, (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), True))
)

def rule_73_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 73
        rule_73(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
