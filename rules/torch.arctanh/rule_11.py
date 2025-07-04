import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the input tensor's values are all very close to 1 or -1, the result may contain inf/nan. (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(Or([And(x < (Select(v["arg1_range"], 1) + 1), Or((And(x > 0.9999, x < 1.0001)), (And(x < -0.9999, x > -1.0001)))) for x in range(6)])) if n else
          Or([And(x < (Select(v["arg1_range"], 1) + 1), Or((And(x > 0.9999, x < 1.0001)), (And(x < -0.9999, x > -1.0001)))) for x in range(6)]))
)

def rule_11_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 11
        rule_11(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_range': arg1['range']}, neg)
