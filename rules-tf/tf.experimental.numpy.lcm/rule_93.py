import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If both tensors have more than 1 dimension, the sum of their dimensions should not be prime numbers (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] > 1, v["arg2_ndim"] > 1), And(And(And((v["arg1_ndim"] + v["arg2_ndim"] != 2), (v["arg1_ndim"] + v["arg2_ndim"] != 3)), (v["arg1_ndim"] + v["arg2_ndim"] != 5)), (v["arg1_ndim"] + v["arg2_ndim"] != 7)), True)) if n else
          If(And(v["arg1_ndim"] > 1, v["arg2_ndim"] > 1), And(And(And((v["arg1_ndim"] + v["arg2_ndim"] != 2), (v["arg1_ndim"] + v["arg2_ndim"] != 3)), (v["arg1_ndim"] + v["arg2_ndim"] != 5)), (v["arg1_ndim"] + v["arg2_ndim"] != 7)), True))
)

def rule_93_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 93
        rule_93(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)
