import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# concrete_function must not be a string with function related name (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(And(And(And((v["arg1_value"] != v["arg2_value"]), (v["arg1_value"] != v["arg3_value"])), (v["arg1_value"] != v["arg4_value"])), (v["arg1_value"] != v["arg5_value"]))) if n else
          And(And(And((v["arg1_value"] != v["arg2_value"]), (v["arg1_value"] != v["arg3_value"])), (v["arg1_value"] != v["arg4_value"])), (v["arg1_value"] != v["arg5_value"])))
)

def rule_93_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, str):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = String('arg3_value')
        arg4_value = String('arg4_value')
        arg5_value = String('arg5_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        solver.add(arg5_value == list_of_string_values_tf.index(arg5))

        # Constraints for rule 93
        rule_93(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
