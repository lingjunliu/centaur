import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if x and y are not scalar, they should not be both of integer and floating type (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(Or((v["arg1_ndim"] == 0), (v["arg2_ndim"] == 0)), True, If(And((Or(Or(Or((v["arg1_dtype"] == 6), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 9))), (Or(Or(Or(Or((v["arg2_dtype"] == 1), (v["arg2_dtype"] == 2)), (v["arg2_dtype"] == 3)), (v["arg2_dtype"] == 4)), (v["arg2_dtype"] == 5)))), False, If(And((Or(Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 9))), (Or(Or(Or(Or((v["arg1_dtype"] == 1), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 4)), (v["arg1_dtype"] == 5)))), False, True)))) if n else
          If(Or((v["arg1_ndim"] == 0), (v["arg2_ndim"] == 0)), True, If(And((Or(Or(Or((v["arg1_dtype"] == 6), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 9))), (Or(Or(Or(Or((v["arg2_dtype"] == 1), (v["arg2_dtype"] == 2)), (v["arg2_dtype"] == 3)), (v["arg2_dtype"] == 4)), (v["arg2_dtype"] == 5)))), False, If(And((Or(Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8)), (v["arg2_dtype"] == 9))), (Or(Or(Or(Or((v["arg1_dtype"] == 1), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 4)), (v["arg1_dtype"] == 5)))), False, True))))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 35
        rule_35(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
