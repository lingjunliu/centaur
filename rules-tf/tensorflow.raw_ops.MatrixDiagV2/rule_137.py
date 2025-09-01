import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If num_cols is defined, then it has to be defined as int32  (Rule 137)

rule_137 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, v["arg1_dtype"] == 3, True)) if n else
          If(v["arg1_ndim"] == 0, v["arg1_dtype"] == 3, True))
)

def rule_137_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 137
        rule_137(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_137(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']}, neg)
