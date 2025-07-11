import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If int type, either v_1 is between [0, (2**64 (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= 0, Or(v["arg1_value"] <= 18446744073709551615, (v["arg1_value"] + 18446744073709551616) >= 0))) if n else
          And(v["arg1_value"] >= 0, Or(v["arg1_value"] <= 18446744073709551615, (v["arg1_value"] + 18446744073709551616) >= 0)))
)

def rule_52_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value']}, neg)
