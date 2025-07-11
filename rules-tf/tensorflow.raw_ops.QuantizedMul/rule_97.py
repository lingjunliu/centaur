import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the output is provided and the dtypes of x and y are the same, their min values must be less or equal to 0  (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(If(And(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 5), v["arg1_value"] == 3), v["arg1_value"] == 2), v["arg1_value"] == 14), v["arg2_dtype"] == v["arg3_dtype"]), And(Select(v["arg2_range"], 0) <= 0, Select(v["arg3_range"], 0) <= 0), False)) if n else
          If(And(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 5), v["arg1_value"] == 3), v["arg1_value"] == 2), v["arg1_value"] == 14), v["arg2_dtype"] == v["arg3_dtype"]), And(Select(v["arg2_range"], 0) <= 0, Select(v["arg3_range"], 0) <= 0), False))
)

def rule_97_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 97
        rule_97(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg3_dtype': arg3_dtype, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg3_dtype': arg3['dtype'], 'arg3_range': arg3['range']}, neg)
