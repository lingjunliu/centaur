import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# axis must be in the range [-ndim(input), ndim(input) - 1] (argmin/argmax axis rule)
# {v_1 : tensor, v_2 : int} |= -ndim(v_1) <= v_2 <= ndim(v_1) - 1

rule_126 = lambda s, v, n=False: (
    s.add(Not(And(
        v["arg2_val"] >= -v["arg1_ndim"],
        v["arg2_val"] <= v["arg1_ndim"] - 1
    )) if n else
    And(
        v["arg2_val"] >= -v["arg1_ndim"],
        v["arg2_val"] <= v["arg1_ndim"] - 1
    ))
)

def rule_126_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (int, np.integer)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_val = Int('arg2_val')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_val == int(arg2))

        # Constraints for axis rule
        rule_126(solver, {'arg1_ndim': arg1_ndim, 'arg2_val': arg2_val})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_126(solver, {'arg1_ndim': arg1['ndim'], 'arg2_val': arg2['val']}, neg)