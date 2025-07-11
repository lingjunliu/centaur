import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# if elements is scalar, require test_elements have the same dtype, else require test_elements have supported dtypes (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_ndim"] == 0), (v["arg1_dtype"] == v["arg2_dtype"]), If((v["arg2_dtype"] == 0), False, Or((v["arg2_dtype"] == 11), (v["arg2_dtype"] == 12))))) if n else
          If((v["arg1_ndim"] == 0), (v["arg1_dtype"] == v["arg2_dtype"]), If((v["arg2_dtype"] == 0), False, Or((v["arg2_dtype"] == 11), (v["arg2_dtype"] == 12)))))
)

def rule_86_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 86
        rule_86(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
