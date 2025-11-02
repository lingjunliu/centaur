import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is true, the dtype of v_2 should be int, otherwise float (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], And(v["arg2_dtype"] <= 5, v["arg2_dtype"] >= 1), And(v["arg2_dtype"] >= 7, v["arg2_dtype"] <= 8))) if n else
          If(v["arg1_value"], And(v["arg2_dtype"] <= 5, v["arg2_dtype"] >= 1), And(v["arg2_dtype"] >= 7, v["arg2_dtype"] <= 8)))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 21
        rule_21(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
