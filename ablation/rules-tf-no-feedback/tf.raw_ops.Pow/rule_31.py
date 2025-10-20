import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If y is an integer tensor, then elements of y must be in the range [-128, 127] for int8, [-32768, 32767] for int16, and so on. (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(((v["arg2_dtype"] == 1)), And(Select(v["arg2_range"], 0) >= -128, Select(v["arg2_range"], 1) <= 127), If(((v["arg2_dtype"] == 2)), And(Select(v["arg2_range"], 0) >= -32768, Select(v["arg2_range"], 1) <= 32767), If(((v["arg2_dtype"] == 3)), And(Select(v["arg2_range"], 0) >= -2147483648, Select(v["arg2_range"], 1) <= 2147483647), If(((v["arg2_dtype"] == 4)), True, True))))) if n else
          If(((v["arg2_dtype"] == 1)), And(Select(v["arg2_range"], 0) >= -128, Select(v["arg2_range"], 1) <= 127), If(((v["arg2_dtype"] == 2)), And(Select(v["arg2_range"], 0) >= -32768, Select(v["arg2_range"], 1) <= 32767), If(((v["arg2_dtype"] == 3)), And(Select(v["arg2_range"], 0) >= -2147483648, Select(v["arg2_range"], 1) <= 2147483647), If(((v["arg2_dtype"] == 4)), True, True)))))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 31
        rule_31(solver, {'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
