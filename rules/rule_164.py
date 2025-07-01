import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if there exists shape in dimension >=0 and <=5 and there exist one dimension of size one, then the data type should be greater than or equal to float 16 (Rule 164)

rule_164 = lambda s, v, n=False: (
    s.add(Not(If((Or([And(i < (If(v["arg1_ndim"] > 5, 5, v["arg1_ndim"] - 1) + 1), Select(v["arg1_shape"], i) >= 1) for i in range(6)])), v["arg1_dtype"] >= 6, False)) if n else
          If((Or([And(i < (If(v["arg1_ndim"] > 5, 5, v["arg1_ndim"] - 1) + 1), Select(v["arg1_shape"], i) >= 1) for i in range(6)])), v["arg1_dtype"] >= 6, False))
)

def rule_164_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 164
        rule_164(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_164(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']}, neg)
