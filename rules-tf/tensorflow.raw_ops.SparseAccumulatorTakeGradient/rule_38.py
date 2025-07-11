import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dtype is qint8 or quint8 etc., the value tensor must be of the same type. (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg1_value"] == 12, v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg2_dtype"] == v["arg1_value"], False)) if n else
          If(Or(Or(Or(Or(v["arg1_value"] == 12, v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg2_dtype"] == v["arg1_value"], False))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
