import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the data format is not NHWC or NCHW or channels_first or channels_last, then the op will throw an error (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_value"] == 33, v["arg1_value"] == 34), v["arg1_value"] == 25), v["arg1_value"] == 24)) if n else
          Or(Or(Or(v["arg1_value"] == 33, v["arg1_value"] == 34), v["arg1_value"] == 25), v["arg1_value"] == 24))
)

def rule_87_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))

        # Constraints for rule 87
        rule_87(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_value': arg1['value']}, neg)
