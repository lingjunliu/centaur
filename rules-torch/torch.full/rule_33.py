import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# size should be a tuple of int and all dimensions in size tuple must be non-negative and less than max int value (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < 2147483647)) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < 2147483647)) for i in range(6)]))
)

def rule_33_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 33
        rule_33(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
