import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# size elements should be non-negative and small, avoiding InvalidArgumentError and ResourceExhaustedError (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, And(And(v["arg1_length"] <= 6, (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < 2000)) for i in range(6)]))), (If(v["arg1_length"] == 1, Select(v["arg1_values"], 0) <= 10000000000000000, (And([Implies(i < (v["arg1_length"] - 1 + 1), If(i > 0, Select(v["arg1_values"], i) <= 10000000000000000 / Select(v["arg1_values"], i - 1), Select(v["arg1_values"], i) <= 10000000000000000)) for i in range(6)]))))), True)) if n else
          If(v["arg1_length"] > 0, And(And(v["arg1_length"] <= 6, (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < 2000)) for i in range(6)]))), (If(v["arg1_length"] == 1, Select(v["arg1_values"], 0) <= 10000000000000000, (And([Implies(i < (v["arg1_length"] - 1 + 1), If(i > 0, Select(v["arg1_values"], i) <= 10000000000000000 / Select(v["arg1_values"], i - 1), Select(v["arg1_values"], i) <= 10000000000000000)) for i in range(6)]))))), True))
)

def rule_30_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 30
        rule_30(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
