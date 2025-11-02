import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is specified and requires_grad is true, and input dtype is not float or complex, dtype must be float or complex (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == True, (And(And(And(v["arg3_dtype"] != 7, v["arg3_dtype"] != 8), v["arg3_dtype"] != 9), v["arg3_dtype"] != 10))), (Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10)), True)) if n else
          If(And(v["arg2_value"] == True, (And(And(And(v["arg3_dtype"] != 7, v["arg3_dtype"] != 8), v["arg3_dtype"] != 9), v["arg3_dtype"] != 10))), (Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10)), True))
)

def rule_28_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
