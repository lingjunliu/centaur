import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor and dims tuple must be dimension compatible such that indices are valid and unique, the tensor should have at most one dimension when the tuple is empty, if tensor is 1D, the permute tuple must be (0 (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == v["arg2_length"], And([Implies(i < (v["arg2_length"] - 1 + 1), And(And(And(And(And(0 <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) < v["arg1_ndim"]), (If(v["arg2_length"] > 1, (And([Implies(i < (v["arg2_length"] - 1 + 1), And([Implies(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) != Select(v["arg2_values"], j)) for j in range(6)])) for i in range(6)])), True))), (If(v["arg2_length"] == 0, v["arg1_ndim"] <= 1, True))), (If(v["arg1_ndim"] == 1, And(v["arg2_length"] == 1, Select(v["arg2_values"], 0) == 0), True))), (If(v["arg1_ndim"] > 1, And((Or([And(p < (1 + 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), p == p * Select(v["arg1_shape"], i)) for i in range(6)])) == (Or([And(q < (1 + 1), (And([Implies(i < (v["arg2_length"] - 1 + 1), q == q * Select(v["arg1_shape"], Select(v["arg2_values"], i))) for i in range(6)]))) for q in range(6)]))) for p in range(6)])), (v["arg1_dtype"] == v["arg1_dtype"])), True)))) for i in range(6)]))) if n else
          And(v["arg1_ndim"] == v["arg2_length"], And([Implies(i < (v["arg2_length"] - 1 + 1), And(And(And(And(And(0 <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) < v["arg1_ndim"]), (If(v["arg2_length"] > 1, (And([Implies(i < (v["arg2_length"] - 1 + 1), And([Implies(j < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) != Select(v["arg2_values"], j)) for j in range(6)])) for i in range(6)])), True))), (If(v["arg2_length"] == 0, v["arg1_ndim"] <= 1, True))), (If(v["arg1_ndim"] == 1, And(v["arg2_length"] == 1, Select(v["arg2_values"], 0) == 0), True))), (If(v["arg1_ndim"] > 1, And((Or([And(p < (1 + 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), p == p * Select(v["arg1_shape"], i)) for i in range(6)])) == (Or([And(q < (1 + 1), (And([Implies(i < (v["arg2_length"] - 1 + 1), q == q * Select(v["arg1_shape"], Select(v["arg2_values"], i))) for i in range(6)]))) for q in range(6)]))) for p in range(6)])), (v["arg1_dtype"] == v["arg1_dtype"])), True)))) for i in range(6)])))
)

def rule_74_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 74
        rule_74(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
