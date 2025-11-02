import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

#  If the list of string element are relu,tanh,sigmoid, the tensor needs to have at least one dimension (Rule 125)

rule_125 = lambda s, v, n=False: (
    s.add(Not(If(And([Implies(i < (v["arg2_length"] - 1 + 1), (Or(Or(Select(v["arg2_values"], i) == 11, Select(v["arg2_values"], i) == 12), Select(v["arg2_values"], i) == 13))) for i in range(6)]), v["arg1_ndim"] >= 1, True)) if n else
          If(And([Implies(i < (v["arg2_length"] - 1 + 1), (Or(Or(Select(v["arg2_values"], i) == 11, Select(v["arg2_values"], i) == 12), Select(v["arg2_values"], i) == 13))) for i in range(6)]), v["arg1_ndim"] >= 1, True))
)

def rule_125_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, str) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 125
        rule_125(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_125(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
