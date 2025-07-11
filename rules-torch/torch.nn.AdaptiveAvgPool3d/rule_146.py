import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check the length of output_size if the tensor is not of dimension 4/5 (Rule 146)

rule_146 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_ndim"] != 4, v["arg1_ndim"] != 5)), v["arg2_length"] > 0, False)) if n else
          If((And(v["arg1_ndim"] != 4, v["arg1_ndim"] != 5)), v["arg2_length"] > 0, False))
)

def rule_146_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 146
        rule_146(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_146(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)
