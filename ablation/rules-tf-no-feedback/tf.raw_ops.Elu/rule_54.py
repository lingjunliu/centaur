import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If feature dtype is valid, but name is not, name has to be none. (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 14), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), (And(And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 6, v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 17), v["arg2_value"] != 18), v["arg2_value"] != 19), v["arg2_value"] != 20), v["arg2_value"] != 21), v["arg2_value"] != 22), v["arg2_value"] != 23), v["arg2_value"] != 24), v["arg2_value"] != 25))), v["arg2_value"] == 6, True)) if n else
          If(And((Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 14), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), (And(And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 6, v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 17), v["arg2_value"] != 18), v["arg2_value"] != 19), v["arg2_value"] != 20), v["arg2_value"] != 21), v["arg2_value"] != 22), v["arg2_value"] != 23), v["arg2_value"] != 24), v["arg2_value"] != 25))), v["arg2_value"] == 6, True))
)

def rule_54_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 54
        rule_54(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
