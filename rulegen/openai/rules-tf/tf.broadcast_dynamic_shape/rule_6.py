import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tuple inputs: non-negative dims and right-aligned broadcast compatibility; suppresses “Incompatible shapes” (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(And([Implies(k < (v["arg1_length"] + v["arg2_length"] + 1), If(k <= v["arg1_length"] - 1, If(k <= v["arg2_length"] - 1, (And(And((Or(Or(Select(v["arg1_values"], v["arg1_length"] - 1 - k) == Select(v["arg2_values"], v["arg2_length"] - 1 - k), Select(v["arg1_values"], v["arg1_length"] - 1 - k) == 1), Select(v["arg2_values"], v["arg2_length"] - 1 - k) == 1)), Select(v["arg1_values"], v["arg1_length"] - 1 - k) >= 0), Select(v["arg2_values"], v["arg2_length"] - 1 - k) >= 0)), Select(v["arg1_values"], v["arg1_length"] - 1 - k) >= 0), If(k <= v["arg2_length"] - 1, Select(v["arg2_values"], v["arg2_length"] - 1 - k) >= 0, True))) for k in range(6)])) if n else
          And([Implies(k < (v["arg1_length"] + v["arg2_length"] + 1), If(k <= v["arg1_length"] - 1, If(k <= v["arg2_length"] - 1, (And(And((Or(Or(Select(v["arg1_values"], v["arg1_length"] - 1 - k) == Select(v["arg2_values"], v["arg2_length"] - 1 - k), Select(v["arg1_values"], v["arg1_length"] - 1 - k) == 1), Select(v["arg2_values"], v["arg2_length"] - 1 - k) == 1)), Select(v["arg1_values"], v["arg1_length"] - 1 - k) >= 0), Select(v["arg2_values"], v["arg2_length"] - 1 - k) >= 0)), Select(v["arg1_values"], v["arg1_length"] - 1 - k) >= 0), If(k <= v["arg2_length"] - 1, Select(v["arg2_values"], v["arg2_length"] - 1 - k) >= 0, True))) for k in range(6)]))
)

def rule_6_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 6
        rule_6(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
