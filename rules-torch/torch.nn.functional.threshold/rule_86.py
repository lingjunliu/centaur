import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# value needs to be able to convert to float16 when dtype of the input is float16 (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 6, Or([And(x < (6.5e4 + 1), v["arg1_value"] == x) for x in range(6)]), False)) if n else
          If(v["arg2_value"] == 6, Or([And(x < (6.5e4 + 1), v["arg1_value"] == x) for x in range(6)]), False))
)

def rule_86_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 86
        rule_86(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
