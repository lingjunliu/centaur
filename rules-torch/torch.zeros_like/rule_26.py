import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is specified and is not None, and requires_grad is specified, dtype must be float or complex if requires_grad is true (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"], v["arg1_value"] != 12), (Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10)), True)) if n else
          If(And(v["arg2_value"], v["arg1_value"] != 12), (Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10)), True))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 26
        rule_26(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
