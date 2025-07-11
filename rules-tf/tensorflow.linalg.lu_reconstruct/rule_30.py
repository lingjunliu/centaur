import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If perm is not "none" and validate_args is true, then the number of dimensions of lower_upper must be one greater than that of perm (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] != 6, v["arg4_value"] == True), v["arg2_ndim"] == v["arg3_ndim"] + 1, False)) if n else
          If(And(v["arg1_value"] != 6, v["arg4_value"] == True), v["arg2_ndim"] == v["arg3_ndim"] + 1, False))
)

def rule_30_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == arg4)

        # Constraints for rule 30
        rule_30(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
