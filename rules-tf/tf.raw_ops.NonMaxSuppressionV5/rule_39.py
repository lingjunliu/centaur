import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# boxes and scores must be float16 or float32 and soft_nms_sigma must be a non-negative float and pad_to_max_output_size should influence max_output_size (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(And(And((Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6))), Select(v["arg2_range"], 0) >= 0.0), (If(v["arg3_value"] == True, Select(v["arg4_range"], 0) > 0, True)))) if n else
          And(And((Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6))), Select(v["arg2_range"], 0) >= 0.0), (If(v["arg3_value"] == True, Select(v["arg4_range"], 0) > 0, True))))
)

def rule_39_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 39
        rule_39(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_range': arg4_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_range': arg4['range']}, neg)
