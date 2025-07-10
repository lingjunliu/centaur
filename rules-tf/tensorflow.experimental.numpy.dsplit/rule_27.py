import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The product of the elements in the indices tuple must be equal to shape(v_2, 2 (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > 0, (And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], j) > 0) for j in range(6)]))), (Select(v["arg2_shape"], 2) == (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1))))) for i in range(6)])) if n else
          Or([And(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > 0, (And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], j) > 0) for j in range(6)]))), (Select(v["arg2_shape"], 2) == (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1))))) for i in range(6)]))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 27
        rule_27(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_shape': arg2['shape']}, neg)
