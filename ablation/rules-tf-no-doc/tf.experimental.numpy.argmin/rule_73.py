import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When axis is not None and keepdims is True, the output dimension is the same as the input dimension (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] != 6, v["arg2_value"]), v["arg3_ndim"] == v["arg4_ndim"], True)) if n else
          If(And(v["arg1_value"] != 6, v["arg2_value"]), v["arg3_ndim"] == v["arg4_ndim"], True))
)

def rule_73_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 73
        rule_73(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg4_ndim': arg4['ndim']}, neg)
