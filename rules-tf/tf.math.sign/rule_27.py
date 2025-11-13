import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# x's dtype must not be bool, int8, int16, uint8, str, or dtype (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((v["arg1_dtype"] != 0), (v["arg1_dtype"] != 1)), (v["arg1_dtype"] != 2)), (v["arg1_dtype"] != 5)), (v["arg1_dtype"] != 11)), (v["arg1_dtype"] != 12))) if n else
          And(And(And(And(And((v["arg1_dtype"] != 0), (v["arg1_dtype"] != 1)), (v["arg1_dtype"] != 2)), (v["arg1_dtype"] != 5)), (v["arg1_dtype"] != 11)), (v["arg1_dtype"] != 12)))
)

def rule_27_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 27
        rule_27(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_dtype': arg1['dtype']}, neg)
