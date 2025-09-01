import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Should have dimension property and elements that can have their type checked  (Rule 114)

rule_114 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (If(0 < 0, 0, 0) + 1), And(And(True, (v["arg1_dtype"] > 0)), (v["arg1_dtype"] < 9))) for i in range(6)])) if n else
          Or([And(i < (If(0 < 0, 0, 0) + 1), And(And(True, (v["arg1_dtype"] > 0)), (v["arg1_dtype"] < 9))) for i in range(6)]))
)

def rule_114_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 114
        rule_114(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_114(solver, {'arg1_dtype': arg1['dtype']}, neg)
