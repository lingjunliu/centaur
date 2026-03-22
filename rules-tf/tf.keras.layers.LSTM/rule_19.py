import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If recurrent_dropout is 0 and unroll is false, and activation is tanh, and recurrent_activation is sigmoid and use_bias is true, then the cuDNN implementation will be used if available, else the default implementation (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(v["arg1_value"] == 0, v["arg2_value"] == False), v["arg3_value"] == 12), v["arg4_value"] == 13), v["arg5_value"] == True), True, True)) if n else
          If(And(And(And(And(v["arg1_value"] == 0, v["arg2_value"] == False), v["arg3_value"] == 12), v["arg4_value"] == 13), v["arg5_value"] == True), True, True))
)

def rule_19_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, str):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = String('arg3_value')
        arg4_value = String('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        solver.add(arg5_value == arg5)

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
