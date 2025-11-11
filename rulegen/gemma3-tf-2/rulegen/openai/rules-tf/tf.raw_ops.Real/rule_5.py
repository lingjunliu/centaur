import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# type compatibility between input complex dtype and Tout (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(And((If(v["arg1_dtype"] == 9, v["arg2_value"] == 7, True)), (If(v["arg1_dtype"] == 10, (Or(v["arg2_value"] == 7, v["arg2_value"] == 8)), True)))) if n else
          And((If(v["arg1_dtype"] == 9, v["arg2_value"] == 7, True)), (If(v["arg1_dtype"] == 10, (Or(v["arg2_value"] == 7, v["arg2_value"] == 8)), True))))
)

def rule_5_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 5
        rule_5(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
