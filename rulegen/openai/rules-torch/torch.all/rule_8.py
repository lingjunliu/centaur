import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tuple dims with keepdim=True preserve ndim; reduced dims become size 1 (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg3_value"] == True, v["arg2_length"] >= 1), v["arg4_ndim"] == v["arg1_ndim"]), And([Implies(i < (v["arg2_length"] - 1 + 1), And((And(-1 * v["arg1_ndim"] <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) <= v["arg1_ndim"] - 1)), And([Implies(j < (v["arg1_ndim"] - 1 + 1), Or((And((Or([And(k < (v["arg2_length"] - 1 + 1), (Or((And(Select(v["arg2_values"], k) >= 0, Select(v["arg2_values"], k) == j)), (And(Select(v["arg2_values"], k) < 0, v["arg1_ndim"] + Select(v["arg2_values"], k) == j))))) for k in range(6)])), Select(v["arg4_shape"], j) == 1)), (And((And([Implies(k < (v["arg2_length"] - 1 + 1), (Or((And(Select(v["arg2_values"], k) >= 0, Select(v["arg2_values"], k) != j)), (And(Select(v["arg2_values"], k) < 0, v["arg1_ndim"] + Select(v["arg2_values"], k) != j))))) for k in range(6)])), Select(v["arg4_shape"], j) == Select(v["arg1_shape"], j))))) for j in range(6)]))) for i in range(6)]))) if n else
          And(And(And(v["arg3_value"] == True, v["arg2_length"] >= 1), v["arg4_ndim"] == v["arg1_ndim"]), And([Implies(i < (v["arg2_length"] - 1 + 1), And((And(-1 * v["arg1_ndim"] <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) <= v["arg1_ndim"] - 1)), And([Implies(j < (v["arg1_ndim"] - 1 + 1), Or((And((Or([And(k < (v["arg2_length"] - 1 + 1), (Or((And(Select(v["arg2_values"], k) >= 0, Select(v["arg2_values"], k) == j)), (And(Select(v["arg2_values"], k) < 0, v["arg1_ndim"] + Select(v["arg2_values"], k) == j))))) for k in range(6)])), Select(v["arg4_shape"], j) == 1)), (And((And([Implies(k < (v["arg2_length"] - 1 + 1), (Or((And(Select(v["arg2_values"], k) >= 0, Select(v["arg2_values"], k) != j)), (And(Select(v["arg2_values"], k) < 0, v["arg1_ndim"] + Select(v["arg2_values"], k) != j))))) for k in range(6)])), Select(v["arg4_shape"], j) == Select(v["arg1_shape"], j))))) for j in range(6)]))) for i in range(6)])))
)

def rule_8_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 8
        rule_8(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape']}, neg)
