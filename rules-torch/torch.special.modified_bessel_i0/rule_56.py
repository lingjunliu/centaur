import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input tensor should not be Half dtype, it should be float16, float32, float64, complex64 or complex128 and finite (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_dtype"] != 6, (Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))), Select(v["arg1_range"], 0) > -1000000000000.0), Select(v["arg1_range"], 1) < 1000000000000.0)) if n else
          And(And(And(v["arg1_dtype"] != 6, (Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))), Select(v["arg1_range"], 0) > -1000000000000.0), Select(v["arg1_range"], 1) < 1000000000000.0))
)

def rule_56_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 56
        rule_56(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']}, neg)
