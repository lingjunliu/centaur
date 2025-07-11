import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If inplace is True, the maximum value that can be represented by the data type T must be large enough to avoid overflows during the subtraction - from <= std::numeric_limits<T>::max( (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, (If(v["arg2_value"] == 7, v["arg1_value"] > -3.4028235e+38, If(v["arg2_value"] == 8, v["arg1_value"] > -1.7976931348623157e+308, False))), False)) if n else
          If(v["arg3_value"] == True, (If(v["arg2_value"] == 7, v["arg1_value"] > -3.4028235e+38, If(v["arg2_value"] == 8, v["arg1_value"] > -1.7976931348623157e+308, False))), False))
)

def rule_35_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 35
        rule_35(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
