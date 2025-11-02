import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# universally exclude a data-format keyword (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (0 + 1), v["arg1_value"] != 25) for i in range(6)])) if n else
          And([Implies(i < (0 + 1), v["arg1_value"] != 25) for i in range(6)]))
)

def rule_13_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 13
        rule_13(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_value': arg1['value']}, neg)
