import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If tensor v_1 has 4 dimensions, and string v_2 equals "channels_first", then integer v_3 must be equal to 1; otherwise, if string v_2 equals "channels_last", then integer v_3 must be equal to 3. (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 4, v["arg2_value"] == 25), v["arg3_value"] == 1, If(And(v["arg1_ndim"] == 4, v["arg2_value"] == 24), v["arg3_value"] == 3, True))) if n else
          If(And(v["arg1_ndim"] == 4, v["arg2_value"] == 25), v["arg3_value"] == 1, If(And(v["arg1_ndim"] == 4, v["arg2_value"] == 24), v["arg3_value"] == 3, True)))
)

def rule_87_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = String('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 87
        rule_87(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
