import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If there are threads running, they all must have unique names (Rule 125)

rule_125 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), And([Implies(j < (v["arg1_length"] - 1 + 1), If(And(Select(v["arg2_values"], i) == True, Select(v["arg2_values"], j) == True), Select(v["arg1_values"], i) != Select(v["arg1_values"], j), True)) for j in range(6)])) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), And([Implies(j < (v["arg1_length"] - 1 + 1), If(And(Select(v["arg2_values"], i) == True, Select(v["arg2_values"], j) == True), Select(v["arg1_values"], i) != Select(v["arg1_values"], j), True)) for j in range(6)])) for i in range(6)]))
)

def rule_125_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, bool) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())
        arg2_values = Array('arg2_values', IntSort(), BoolSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 125
        rule_125(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_125(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_values': arg2['values']}, neg)
