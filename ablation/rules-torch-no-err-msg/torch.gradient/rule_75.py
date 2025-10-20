import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The dim list must contain unique values, and they must be valid dimensions of input tensor, but must be positive (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_length"] > 0, (And([Implies(i < (v["arg1_length"] - 1 + 1), And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) != Select(v["arg1_values"], j)) for j in range(6)])) for i in range(6)]))), (And([Implies(i < (v["arg1_length"] - 1 + 1), And(0 <= Select(v["arg1_values"], i), Select(v["arg1_values"], i) < v["arg2_ndim"])) for i in range(6)])))) if n else
          And(And(v["arg1_length"] > 0, (And([Implies(i < (v["arg1_length"] - 1 + 1), And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) != Select(v["arg1_values"], j)) for j in range(6)])) for i in range(6)]))), (And([Implies(i < (v["arg1_length"] - 1 + 1), And(0 <= Select(v["arg1_values"], i), Select(v["arg1_values"], i) < v["arg2_ndim"])) for i in range(6)]))))
)

def rule_75_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 75
        rule_75(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim']}, neg)
