import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# pooling ratio should be valid for fractional max pooling (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(And(And(Select(v["arg1_values"], 0) == 1.0, Select(v["arg1_values"], 3) == 1.0), And([Implies(i < (2 + 1), And(Select(v["arg1_values"], i) >= 1.0, v["arg2_ndim"] == 4)) for i in range(6)]))) if n else
          And(And(Select(v["arg1_values"], 0) == 1.0, Select(v["arg1_values"], 3) == 1.0), And([Implies(i < (2 + 1), And(Select(v["arg1_values"], i) >= 1.0, v["arg2_ndim"] == 4)) for i in range(6)])))
)

def rule_2_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), RealSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 2
        rule_2(solver, {'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim']}, neg)
