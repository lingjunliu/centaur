import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# When the input tensor is an integer type and no output tensor is specified the resulting tensor's dtype should be float or complex  (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), Or((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))), False)) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), Or((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(9 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))), False))
)

def rule_65_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 65
        rule_65(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_dtype': arg1['dtype']}, neg)
