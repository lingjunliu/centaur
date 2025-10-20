import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if replace complex is false and the given type is different, then check if the dtype needs to be one of the approved casting strings (Rule 124)

rule_124 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, v["arg3_dtype"] != v["arg2_value"]), Or(Or(Or(Or(Or(v["arg4_value"] == 0, v["arg4_value"] == 1), v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5), True)) if n else
          If(And(v["arg1_value"] == False, v["arg3_dtype"] != v["arg2_value"]), Or(Or(Or(Or(Or(v["arg4_value"] == 0, v["arg4_value"] == 1), v["arg4_value"] == 2), v["arg4_value"] == 3), v["arg4_value"] == 4), v["arg4_value"] == 5), True))
)

def rule_124_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = String('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 124
        rule_124(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)
