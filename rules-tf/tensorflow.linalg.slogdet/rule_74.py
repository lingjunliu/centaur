import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Valid shape requires ndim >=2 for floats (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg1_ndim"] >= 2, True)) if n else
          If((Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)), v["arg1_ndim"] >= 2, True))
)

def rule_74_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 74
        rule_74(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
