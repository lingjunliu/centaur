import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# boxes tensor elements should be of float type. (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], 1) - 1 + 1), And([Implies(k < (3 + 1), (v["arg1_dtype"] == 7)) for k in range(6)])) for j in range(6)])) for i in range(6)])) if n else
          And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), And([Implies(j < (Select(v["arg1_shape"], 1) - 1 + 1), And([Implies(k < (3 + 1), (v["arg1_dtype"] == 7)) for k in range(6)])) for j in range(6)])) for i in range(6)]))
)

def rule_25_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 25
        rule_25(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']}, neg)
