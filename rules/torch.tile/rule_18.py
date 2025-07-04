import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The product of `dims` must be representable as an int64 (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) > 0, Or([And(p < (9223372036854775807 + 1), If(v["arg1_length"] == 0, p == 1, (And(p == 1, (And([Implies(i < (v["arg1_length"] - 1 + 1), p == p * Select(v["arg1_values"], i)) for i in range(6)])))))) for p in range(6)]))) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) > 0, Or([And(p < (9223372036854775807 + 1), If(v["arg1_length"] == 0, p == 1, (And(p == 1, (And([Implies(i < (v["arg1_length"] - 1 + 1), p == p * Select(v["arg1_values"], i)) for i in range(6)])))))) for p in range(6)]))) for i in range(6)]))
)

def rule_18_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 18
        rule_18(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
