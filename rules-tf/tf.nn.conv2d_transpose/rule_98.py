import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Dilations values in the batch and depth dimension must be 1 when the tensor is 4D. (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg2_ndim"] == 4), (v["arg3_value"] == 33)), (And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 3) == 1)), If(And((v["arg2_ndim"] == 4), (v["arg3_value"] == 34)), (And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 1) == 1)), True))) if n else
          If(And((v["arg2_ndim"] == 4), (v["arg3_value"] == 33)), (And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 3) == 1)), If(And((v["arg2_ndim"] == 4), (v["arg3_value"] == 34)), (And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 1) == 1)), True)))
)

def rule_98_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 98
        rule_98(solver, {'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
