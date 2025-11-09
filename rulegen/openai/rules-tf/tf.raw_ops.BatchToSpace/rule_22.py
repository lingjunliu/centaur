import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# height and width factors exist that are multiples of block_size (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(Or([And(h < (Select(v["arg1_shape"], 1) * v["arg2_value"] + 1), And(And(h % v["arg2_value"] == 0, Select(v["arg1_shape"], 1) == h / v["arg2_value"]), Or([And(w < (Select(v["arg1_shape"], 2) * v["arg2_value"] + 1), And(w % v["arg2_value"] == 0, Select(v["arg1_shape"], 2) == w / v["arg2_value"])) for w in range(6)]))) for h in range(6)])) if n else
          Or([And(h < (Select(v["arg1_shape"], 1) * v["arg2_value"] + 1), And(And(h % v["arg2_value"] == 0, Select(v["arg1_shape"], 1) == h / v["arg2_value"]), Or([And(w < (Select(v["arg1_shape"], 2) * v["arg2_value"] + 1), And(w % v["arg2_value"] == 0, Select(v["arg1_shape"], 2) == w / v["arg2_value"])) for w in range(6)]))) for h in range(6)]))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 22
        rule_22(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
