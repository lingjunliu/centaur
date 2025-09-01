import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if size_average or reduce are specified as bool, then reduction must not be specified (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_value"] == none, v["arg1_value"] == 6)), Or((Or(v["arg2_value"] == none, v["arg2_value"] == 6)), (v["arg3_value"] == 6)))) if n else
          And((Or(v["arg1_value"] == none, v["arg1_value"] == 6)), Or((Or(v["arg2_value"] == none, v["arg2_value"] == 6)), (v["arg3_value"] == 6))))
)

def rule_41_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, bool) or isinstance(arg2, str)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))

        # Constraints for rule 41
        rule_41(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
