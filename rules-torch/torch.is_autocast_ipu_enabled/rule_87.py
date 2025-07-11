import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Tuple of bools all value needs to be true if dim bigger then 3. (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 3, (And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == True) for i in range(6)])), False)) if n else
          If(v["arg2_ndim"] > 3, (And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == True) for i in range(6)])), False))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, bool) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), BoolSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 87
        rule_87(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim']}, neg)
