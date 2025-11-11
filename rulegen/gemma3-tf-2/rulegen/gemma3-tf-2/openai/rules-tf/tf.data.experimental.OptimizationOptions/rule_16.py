import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Invalid to disable defaults and all specific optimizations: v_1=apply_default_optimizations; v_2=noop_elimination; v_3=map_fusion; v_4=map_and_batch_fusion; v_5=map_parallelization; v_6=shuffle_and_repeat_fusion; v_7=filter_with_random_uniform_fusion; v_8=hoist_random_uniform; v_9=map_vectorization; v_10=filter_fusion (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If((And(And(And(And(And(And(And(And(And(v["arg1_value"] == False, v["arg2_value"] == False), v["arg3_value"] == False), v["arg4_value"] == False), v["arg5_value"] == False), v["arg6_value"] == False), v["arg7_value"] == False), v["arg8_value"] == False), v["arg9_value"] == False), v["arg10_value"] == False)), False, True)) if n else
          If((And(And(And(And(And(And(And(And(And(v["arg1_value"] == False, v["arg2_value"] == False), v["arg3_value"] == False), v["arg4_value"] == False), v["arg5_value"] == False), v["arg6_value"] == False), v["arg7_value"] == False), v["arg8_value"] == False), v["arg9_value"] == False), v["arg10_value"] == False)), False, True))
)

def rule_16_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))
    arg9 = next(iter(arg9.values()))
    arg10 = next(iter(arg10.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, bool):
            return False
        if not isinstance(arg6, bool):
            return False
        if not isinstance(arg7, bool):
            return False
        if not isinstance(arg8, bool):
            return False
        if not isinstance(arg9, bool):
            return False
        if not isinstance(arg10, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Bool('arg5_value')
        arg6_value = Bool('arg6_value')
        arg7_value = Bool('arg7_value')
        arg8_value = Bool('arg8_value')
        arg9_value = Bool('arg9_value')
        arg10_value = Bool('arg10_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == arg6)
        solver.add(arg7_value == arg7)
        solver.add(arg8_value == arg8)
        solver.add(arg9_value == arg9)
        solver.add(arg10_value == arg10)

        # Constraints for rule 16
        rule_16(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value, 'arg7_value': arg7_value, 'arg8_value': arg8_value, 'arg9_value': arg9_value, 'arg10_value': arg10_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value'], 'arg7_value': arg7['value'], 'arg8_value': arg8['value'], 'arg9_value': arg9['value'], 'arg10_value': arg10['value']}, neg)
