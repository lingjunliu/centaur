import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Structure is a tuple and the elements are positive floats and greater than 0.1 and all elements are different. (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] > 0, And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > 0, Select(v["arg1_values"], i) > 0.1), (If(v["arg1_length"] > 1, And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) != Select(v["arg1_values"], j)) for j in range(6)]), False)))) for i in range(6)]))) if n else
          And(v["arg1_length"] > 0, And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > 0, Select(v["arg1_values"], i) > 0.1), (If(v["arg1_length"] > 1, And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) != Select(v["arg1_values"], j)) for j in range(6)]), False)))) for i in range(6)])))
)

def rule_72_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 72
        rule_72(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)
