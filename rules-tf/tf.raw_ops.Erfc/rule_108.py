import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if x has ndim 0 or 1, the magnitude must be less than 10, a somewhat reasonable limit for erfc calculations (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] <= 1, And(Select(v["arg1_range"], 1) < 10, Select(v["arg1_range"], 0) > -10), True)) if n else
          If(v["arg1_ndim"] <= 1, And(Select(v["arg1_range"], 1) < 10, Select(v["arg1_range"], 0) > -10), True))
)

def rule_108_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 108
        rule_108(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range']}, neg)
