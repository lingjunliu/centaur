import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# ∀ x ∈ [0,1] : if name is causal, then name must equal linear (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(And([Implies(x < (1 + 1), (If(v["arg1_value"] == 23, v["arg1_value"] == 20, True))) for x in range(6)])) if n else
          And([Implies(x < (1 + 1), (If(v["arg1_value"] == 23, v["arg1_value"] == 20, True))) for x in range(6)]))
)

def rule_62_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))

        # Constraints for rule 62
        rule_62(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_value': arg1['value']}, neg)
