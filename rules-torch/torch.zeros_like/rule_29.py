import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If requires_grad is True, and input is not float or complex, then dtype must be specified and it must be float or complex (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"], (And(And(And(v["arg2_dtype"] != 7, v["arg2_dtype"] != 8), v["arg2_dtype"] != 9), v["arg2_dtype"] != 10))), (Or(Or(Or(v["arg3_value"] == 7, v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10)), True)) if n else
          If(And(v["arg1_value"], (And(And(And(v["arg2_dtype"] != 7, v["arg2_dtype"] != 8), v["arg2_dtype"] != 9), v["arg2_dtype"] != 10))), (Or(Or(Or(v["arg3_value"] == 7, v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10)), True))
)

def rule_29_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 29
        rule_29(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
