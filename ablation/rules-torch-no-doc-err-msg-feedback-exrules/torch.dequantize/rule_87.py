import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if dtype is specified, it should be float (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 0, Or(Or(Or((v["arg1_value"] == 6), (v["arg1_value"] == 7)), (v["arg1_value"] == 8)), (v["arg1_value"] == 12)), True)) if n else
          If(v["arg1_value"] != 0, Or(Or(Or((v["arg1_value"] == 6), (v["arg1_value"] == 7)), (v["arg1_value"] == 8)), (v["arg1_value"] == 12)), True))
)

def rule_87_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 87
        rule_87(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_value': arg1['value']}, neg)
