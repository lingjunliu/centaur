import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# tuple of floats, check if all elements are greater or equal 0 and min element greater than 5 or tuple is empty or all values are the same (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And((And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) >= 0.0) for i in range(6)])), (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 5.0) for i in range(6)])))), v["arg1_length"] == 0), (And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], 0) == Select(v["arg1_values"], i)) for i in range(6)])))) if n else
          Or(Or((And((And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) >= 0.0) for i in range(6)])), (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 5.0) for i in range(6)])))), v["arg1_length"] == 0), (And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], 0) == Select(v["arg1_values"], i)) for i in range(6)]))))
)

def rule_87_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 87
        rule_87(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
