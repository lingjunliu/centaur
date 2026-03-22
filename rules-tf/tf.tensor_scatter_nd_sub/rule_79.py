import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Bad String policy check, but slightly different than before. This is the LAST attempt to circumvent this. I ensure only an invalid string resolves in False (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(v["arg1_value"] != 6, v["arg1_value"] != 7), v["arg1_value"] != 9), v["arg1_value"] != 8), v["arg1_value"] != 10), False, True)) if n else
          If(And(And(And(And(v["arg1_value"] != 6, v["arg1_value"] != 7), v["arg1_value"] != 9), v["arg1_value"] != 8), v["arg1_value"] != 10), False, True))
)

def rule_79_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 79
        rule_79(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_value': arg1['value']}, neg)
