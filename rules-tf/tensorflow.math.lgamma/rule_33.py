import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# At least one element in tensor must be in range -5, 5 (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not((Or([And(x < (Select(v["arg1_range"], 1) + 1), And(-5 <= x, x <= 5)) for x in range(6)]))) if n else
          (Or([And(x < (Select(v["arg1_range"], 1) + 1), And(-5 <= x, x <= 5)) for x in range(6)])))
)

def rule_33_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 33
        rule_33(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_range': arg1['range']}, neg)
