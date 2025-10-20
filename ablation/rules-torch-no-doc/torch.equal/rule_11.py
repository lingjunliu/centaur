import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If tensor v_1 has at least one element equal to 0 then v_2 must also have at least one element equal to 0 if they are to be equal (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(And((Or([And(x < (Select(v["arg1_range"], 1) + 1), x == 0) for x in range(6)])), (Or([And(y < (Select(v["arg2_range"], 1) + 1), y == 0) for y in range(6)])))) if n else
          And((Or([And(x < (Select(v["arg1_range"], 1) + 1), x == 0) for x in range(6)])), (Or([And(y < (Select(v["arg2_range"], 1) + 1), y == 0) for y in range(6)]))))
)

def rule_11_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 11
        rule_11(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range']}, neg)
