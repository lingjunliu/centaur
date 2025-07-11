import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When the fill value is not specified, default fill value for different dtypes should be in a valid range (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, (And(-128 <= 0, 0 <= 127)), If(v["arg1_dtype"] == 2, (And(-32768 <= 0, 0 <= 32767)), If(v["arg1_dtype"] == 3, (And(-2147483648 <= 0, 0 <= 2147483647)), If(v["arg1_dtype"] == 4, (And(-9223372036854775808 <= 0, 0 <= 9223372036854775807)), If(v["arg1_dtype"] == 5, (And(0 >= 0, 0 <= 255)), If(v["arg1_dtype"] == 6, (And(-65504 <= 0, 0 <= 65504)), False))))))) if n else
          If(v["arg1_dtype"] == 1, (And(-128 <= 0, 0 <= 127)), If(v["arg1_dtype"] == 2, (And(-32768 <= 0, 0 <= 32767)), If(v["arg1_dtype"] == 3, (And(-2147483648 <= 0, 0 <= 2147483647)), If(v["arg1_dtype"] == 4, (And(-9223372036854775808 <= 0, 0 <= 9223372036854775807)), If(v["arg1_dtype"] == 5, (And(0 >= 0, 0 <= 255)), If(v["arg1_dtype"] == 6, (And(-65504 <= 0, 0 <= 65504)), False)))))))
)

def rule_46_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype']}, neg)
