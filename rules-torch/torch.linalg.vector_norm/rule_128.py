import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The specified dtype, when not None, must be convertible from the input dtype AND out type and in type must align (Rule 128)

rule_128 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] != None, (If(v["arg2_value"] == 6, (And(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg3_dtype"] == 6)), If(v["arg2_value"] == 7, (And(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg3_dtype"] == 7)), If(v["arg2_value"] == 8, (And(v["arg1_dtype"] == 8, v["arg3_dtype"] == 8)), If(v["arg2_value"] == 9, (And(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg3_dtype"] == 9)), If(v["arg2_value"] == 10, (And(v["arg1_dtype"] == 10, v["arg3_dtype"] == 10)), False)))))), True)) if n else
          If(v["arg2_value"] != None, (If(v["arg2_value"] == 6, (And(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg3_dtype"] == 6)), If(v["arg2_value"] == 7, (And(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg3_dtype"] == 7)), If(v["arg2_value"] == 8, (And(v["arg1_dtype"] == 8, v["arg3_dtype"] == 8)), If(v["arg2_value"] == 9, (And(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg3_dtype"] == 9)), If(v["arg2_value"] == 10, (And(v["arg1_dtype"] == 10, v["arg3_dtype"] == 10)), False)))))), True))
)

def rule_128_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 128
        rule_128(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_128(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
