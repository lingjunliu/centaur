import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The bit width of y must be smaller or equal to the bit width of x. Assuming int8 is 8, int16 is 16, int32 is 32, int64 is 64, unit8 is 8, unit16 is 16, unit32 is 32, unit64 is 64. (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, False, If(v["arg2_dtype"] == 3, False, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, False, If(v["arg2_dtype"] == 7, False, False))))))), If(v["arg1_dtype"] == 2, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, True, If(v["arg2_dtype"] == 3, False, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, True, If(v["arg2_dtype"] == 7, False, False))))))), If(v["arg1_dtype"] == 3, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, True, If(v["arg2_dtype"] == 3, True, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, True, If(v["arg2_dtype"] == 7, True, False))))))), If(v["arg1_dtype"] == 4, True, If(v["arg1_dtype"] == 5, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, True, If(v["arg2_dtype"] == 3, True, If(v["arg2_dtype"] == 4, True, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, True, If(v["arg2_dtype"] == 7, True, True))))))), True)))))) if n else
          If(v["arg1_dtype"] == 1, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, False, If(v["arg2_dtype"] == 3, False, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, False, If(v["arg2_dtype"] == 7, False, False))))))), If(v["arg1_dtype"] == 2, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, True, If(v["arg2_dtype"] == 3, False, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, True, If(v["arg2_dtype"] == 7, False, False))))))), If(v["arg1_dtype"] == 3, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, True, If(v["arg2_dtype"] == 3, True, If(v["arg2_dtype"] == 4, False, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, True, If(v["arg2_dtype"] == 7, True, False))))))), If(v["arg1_dtype"] == 4, True, If(v["arg1_dtype"] == 5, If(v["arg2_dtype"] == 1, True, If(v["arg2_dtype"] == 2, True, If(v["arg2_dtype"] == 3, True, If(v["arg2_dtype"] == 4, True, If(v["arg2_dtype"] == 5, True, If(v["arg2_dtype"] == 6, True, If(v["arg2_dtype"] == 7, True, True))))))), True))))))
)

def rule_66_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 66
        rule_66(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
