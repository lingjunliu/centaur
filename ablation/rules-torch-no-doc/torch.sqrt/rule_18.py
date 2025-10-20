import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If Short is the target dtype, input should be Short or a Float/Complex dtype, and max input value less than 32768 for avoiding data loss (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 2, (Or(Or(Or(Or(Or(v["arg1_dtype"] == 2, (And(v["arg1_dtype"] == 6, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 7, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 8, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 9, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 10, Select(v["arg1_range"], 1) < 32768)))), True)) if n else
          If(v["arg2_value"] == 2, (Or(Or(Or(Or(Or(v["arg1_dtype"] == 2, (And(v["arg1_dtype"] == 6, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 7, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 8, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 9, Select(v["arg1_range"], 1) < 32768))), (And(v["arg1_dtype"] == 10, Select(v["arg1_range"], 1) < 32768)))), True))
)

def rule_18_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 18
        rule_18(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
