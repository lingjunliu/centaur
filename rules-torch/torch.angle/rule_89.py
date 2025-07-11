import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If out tensor is specified the output tensor's dtype must be float or complex using exists, make robust, add parenthesis, different quant_expr structure (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, Or([And(x < (10 + 1), Or((And(v["arg1_dtype"] == x, x <= 8)), (And(v["arg1_dtype"] == x, x >= 9)))) for x in range(6)]), False)) if n else
          If(v["arg1_ndim"] > 0, Or([And(x < (10 + 1), Or((And(v["arg1_dtype"] == x, x <= 8)), (And(v["arg1_dtype"] == x, x >= 9)))) for x in range(6)]), False))
)

def rule_89_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 89
        rule_89(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']}, neg)
