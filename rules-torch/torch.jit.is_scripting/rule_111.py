import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if a tensor v_1 has ndim = 0 and an int v_2 is equal to 1 then ∃x ∈ [0,1] such that x is 1 and a list v3 of boolean has a length of 1 (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 0, v["arg2_value"] == 1), Or([And(x < (1 + 1), And(x == 1, v["arg3_length"] == 1)) for x in range(6)]), False)) if n else
          If(And(v["arg1_ndim"] == 0, v["arg2_value"] == 1), Or([And(x < (1 + 1), And(x == 1, v["arg3_length"] == 1)) for x in range(6)]), False))
)

def rule_111_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, list) and all(isinstance(e, bool) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 111
        rule_111(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
