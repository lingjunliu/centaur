import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Elements of the dest tuple should be within the allowed range and destination tuple have unique elements (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg2_length"] - 1 + 1), And(And((0 - v["arg1_ndim"]) <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) < v["arg1_ndim"]), And([Implies(k < (v["arg2_length"] - 1 + 1), And([Implies(l < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], k) != Select(v["arg2_values"], l)) for l in range(6)])) for k in range(6)]))) for i in range(6)])) if n else
          And([Implies(i < (v["arg2_length"] - 1 + 1), And(And((0 - v["arg1_ndim"]) <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) < v["arg1_ndim"]), And([Implies(k < (v["arg2_length"] - 1 + 1), And([Implies(l < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], k) != Select(v["arg2_values"], l)) for l in range(6)])) for k in range(6)]))) for i in range(6)]))
)

def rule_102_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 102
        rule_102(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
