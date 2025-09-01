import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# indices_or_sections is list and all the elements sum is less than shape(v_1, 1 (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg2_length"] == 0), (If(v["arg1_ndim"] > 1, (Or([And(sum_ < (Select(v["arg1_shape"], 1) + 1), And((And([Implies(i < (v["arg2_length"] - 1 + 1), sum_ == sum_ - Select(v["arg2_values"], i)) for i in range(6)])), (sum_ >= 0))) for sum_ in range(6)])), True)))) if n else
          Or((v["arg2_length"] == 0), (If(v["arg1_ndim"] > 1, (Or([And(sum_ < (Select(v["arg1_shape"], 1) + 1), And((And([Implies(i < (v["arg2_length"] - 1 + 1), sum_ == sum_ - Select(v["arg2_values"], i)) for i in range(6)])), (sum_ >= 0))) for sum_ in range(6)])), True))))
)

def rule_40_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 40
        rule_40(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
