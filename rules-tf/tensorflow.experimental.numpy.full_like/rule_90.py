import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype_ is specified as a boolean, then fill_value can only be a boolean 0 or 1 value, or an integer 0 or 1, or a float 0.0, 1.0 (Rule 90)

rule_90 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, Or(Or((v["arg2_dtype"] == 0), (And(And((And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 5)), (Or(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 0) == 1))), (Or(Select(v["arg2_range"], 1) == 0, Select(v["arg2_range"], 1) == 1))))), (And(And((And(v["arg2_dtype"] >= 6, v["arg2_dtype"] <= 8)), (Or(Select(v["arg2_range"], 0) == 0.0, Select(v["arg2_range"], 0) == 1.0))), (Or(Select(v["arg2_range"], 1) == 0.0, Select(v["arg2_range"], 1) == 1.0))))), True)) if n else
          If(v["arg1_value"] == 0, Or(Or((v["arg2_dtype"] == 0), (And(And((And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 5)), (Or(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 0) == 1))), (Or(Select(v["arg2_range"], 1) == 0, Select(v["arg2_range"], 1) == 1))))), (And(And((And(v["arg2_dtype"] >= 6, v["arg2_dtype"] <= 8)), (Or(Select(v["arg2_range"], 0) == 0.0, Select(v["arg2_range"], 0) == 1.0))), (Or(Select(v["arg2_range"], 1) == 0.0, Select(v["arg2_range"], 1) == 1.0))))), True))
)

def rule_90_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 90
        rule_90(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_90(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
