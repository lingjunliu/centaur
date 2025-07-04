import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# start and end should be within the range of supported float or complex values for a given dtype (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 6, And(v["arg1_value"] >= -65504.0, v["arg1_value"] <= 65504.0), If(v["arg2_value"] == 7, And(v["arg1_value"] >= -3.4028235e+38, v["arg1_value"] <= 3.4028235e+38), If(v["arg2_value"] == 8, And(v["arg1_value"] >= -1.7976931348623157e+308, v["arg1_value"] <= 1.7976931348623157e+308), If(v["arg2_value"] == 9, And(v["arg1_value"] >= -3.4028235e+38, v["arg1_value"] <= 3.4028235e+38), If(v["arg2_value"] == 10, And(v["arg1_value"] >= -1.7976931348623157e+308, v["arg1_value"] <= 1.7976931348623157e+308), False)))))) if n else
          If(v["arg2_value"] == 6, And(v["arg1_value"] >= -65504.0, v["arg1_value"] <= 65504.0), If(v["arg2_value"] == 7, And(v["arg1_value"] >= -3.4028235e+38, v["arg1_value"] <= 3.4028235e+38), If(v["arg2_value"] == 8, And(v["arg1_value"] >= -1.7976931348623157e+308, v["arg1_value"] <= 1.7976931348623157e+308), If(v["arg2_value"] == 9, And(v["arg1_value"] >= -3.4028235e+38, v["arg1_value"] <= 3.4028235e+38), If(v["arg2_value"] == 10, And(v["arg1_value"] >= -1.7976931348623157e+308, v["arg1_value"] <= 1.7976931348623157e+308), False))))))
)

def rule_10_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 10
        rule_10(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
