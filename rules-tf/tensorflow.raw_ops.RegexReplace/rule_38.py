import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The number of dimensions of pattern and rewrite can't be greater than the number of dimensions of input (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_ndim"] <= v["arg1_ndim"], v["arg3_ndim"] <= v["arg1_ndim"])) if n else
          And(v["arg2_ndim"] <= v["arg1_ndim"], v["arg3_ndim"] <= v["arg1_ndim"]))
)

def rule_38_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 38
        rule_38(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim']}, neg)
