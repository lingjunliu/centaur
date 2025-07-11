import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input should not be ComplexDouble, or if it is, output should be specified as Complex128. Also, prevent Complex inputs if the output is Bool (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_dtype"] != 10), Or((Or(Or((v["arg1_dtype"] < 9), (v["arg1_dtype"] > 10)), (v["arg2_dtype"] != 0))), (And(v["arg1_dtype"] == 10, v["arg2_dtype"] == 11))))) if n else
          And((v["arg1_dtype"] != 10), Or((Or(Or((v["arg1_dtype"] < 9), (v["arg1_dtype"] > 10)), (v["arg2_dtype"] != 0))), (And(v["arg1_dtype"] == 10, v["arg2_dtype"] == 11)))))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 39
        rule_39(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
