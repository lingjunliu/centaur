import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype is float32 or float64, start and step must fit into corresponding float range (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 7, (And(And(And(v["arg1_value"] < 3.4028235e+38, v["arg1_value"] > -3.4028235e+38), v["arg2_value"] < 3.4028235e+38), v["arg2_value"] > -3.4028235e+38)), If(v["arg3_value"] == 8, (And(And(And(v["arg1_value"] < 1.7976931348623157e+308, v["arg1_value"] > -1.7976931348623157e+308), v["arg2_value"] < 1.7976931348623157e+308), v["arg2_value"] > -1.7976931348623157e+308)), True))) if n else
          If(v["arg3_value"] == 7, (And(And(And(v["arg1_value"] < 3.4028235e+38, v["arg1_value"] > -3.4028235e+38), v["arg2_value"] < 3.4028235e+38), v["arg2_value"] > -3.4028235e+38)), If(v["arg3_value"] == 8, (And(And(And(v["arg1_value"] < 1.7976931348623157e+308, v["arg1_value"] > -1.7976931348623157e+308), v["arg2_value"] < 1.7976931348623157e+308), v["arg2_value"] > -1.7976931348623157e+308)), True)))
)

def rule_79_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 79
        rule_79(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
