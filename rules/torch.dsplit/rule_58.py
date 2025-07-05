import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# indices_or_sections must be non-negative and not a very large number or smaller than -1 (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > -1, v["arg1_value"] < 2147483647), v["arg1_value"] != -5425825916322441129)) if n else
          And(And(v["arg1_value"] > -1, v["arg1_value"] < 2147483647), v["arg1_value"] != -5425825916322441129))
)

def rule_58_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value']}, neg)
