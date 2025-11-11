import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Graph.add_to_collections with tuple of names: numeric aggregations imply numeric tensor (Rule 3)

rule_3 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] >= 1, If(And([Implies(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Select(v["arg1_values"], i) == 7, Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10)) for i in range(6)]), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), True))) if n else
          And(v["arg1_length"] >= 1, If(And([Implies(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Select(v["arg1_values"], i) == 7, Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10)) for i in range(6)]), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), True)))
)

def rule_3_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, str) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 3
        rule_3(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_3(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_dtype': arg2['dtype']}, neg)
