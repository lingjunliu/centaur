import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# To avoid mish_cpu error for 'Char' type, the dtype of input tensor v_1 must be numerical when v_2 (inplace (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == False, Or(Or((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5)), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))), (And(10 <= v["arg1_dtype"], v["arg1_dtype"] <= 11))), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)))) if n else
          If(v["arg2_value"] == False, Or(Or((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5)), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))), (And(10 <= v["arg1_dtype"], v["arg1_dtype"] <= 11))), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 26
        rule_26(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
