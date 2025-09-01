import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If bias is False, the input dtype and the ConvTranspose3d layer's dtype must be in the same group (either int/uint or float/complex (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == False, Or((And(And(And(v["arg3_dtype"] >= 1, v["arg3_dtype"] <= 6), v["arg2_value"] >= 1), v["arg2_value"] <= 6)), (And(And(And(v["arg3_dtype"] > 6, v["arg2_value"] <= 11), v["arg3_dtype"] > 6), v["arg2_value"] <= 11))), True)) if n else
          If(v["arg1_value"] == False, Or((And(And(And(v["arg3_dtype"] >= 1, v["arg3_dtype"] <= 6), v["arg2_value"] >= 1), v["arg2_value"] <= 6)), (And(And(And(v["arg3_dtype"] > 6, v["arg2_value"] <= 11), v["arg3_dtype"] > 6), v["arg2_value"] <= 11))), True))
)

def rule_40_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 40
        rule_40(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
