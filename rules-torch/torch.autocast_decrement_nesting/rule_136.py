import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# requires 3 values, v1 a float between -1 and 1, v2 a list of boolean, which can not be empty and should contain a single False value if v1 is smaller than 0 and a single True value if v1 is bigger than 0  (Rule 136)

rule_136 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] > -1, v["arg1_value"] < 1), v["arg2_length"] > 0), (If(v["arg1_value"] < 0, Or([And(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) == False, And([Implies(j < (v["arg2_length"] - 1 + 1), If(i != j, Select(v["arg2_values"], j) == True, Or([And(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) == True, And([Implies(j < (v["arg2_length"] - 1 + 1), If(i != j, Select(v["arg2_values"], j) == False, False)) for j in range(6)]))) for i in range(6)]))) for j in range(6)]))) for i in range(6)]), False)))) if n else
          And(And(And(v["arg1_value"] > -1, v["arg1_value"] < 1), v["arg2_length"] > 0), (If(v["arg1_value"] < 0, Or([And(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) == False, And([Implies(j < (v["arg2_length"] - 1 + 1), If(i != j, Select(v["arg2_values"], j) == True, Or([And(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) == True, And([Implies(j < (v["arg2_length"] - 1 + 1), If(i != j, Select(v["arg2_values"], j) == False, False)) for j in range(6)]))) for i in range(6)]))) for j in range(6)]))) for i in range(6)]), False))))
)

def rule_136_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, bool) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), BoolSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 136
        rule_136(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_136(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
