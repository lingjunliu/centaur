import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Valid single integer source and destination, and all conditions for valid tuple source/dest (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((0 - v["arg5_ndim"]) <= v["arg1_value"], v["arg1_value"] < v["arg5_ndim"]), (0 - v["arg5_ndim"]) <= v["arg2_value"]), v["arg2_value"] < v["arg5_ndim"]), v["arg3_length"] == v["arg4_length"]), And([Implies(i < (v["arg3_length"] - 1 + 1), And(And((0 - v["arg5_ndim"]) <= Select(v["arg3_values"], i), Select(v["arg3_values"], i) < v["arg5_ndim"]), And([Implies(j < (v["arg4_length"] - 1 + 1), And(And(And((0 - v["arg5_ndim"]) <= Select(v["arg4_values"], j), Select(v["arg4_values"], j) < v["arg5_ndim"]), (And([Implies(k < (v["arg3_length"] - 1 + 1), And([Implies(l < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], k) != Select(v["arg3_values"], l)) for l in range(6)])) for k in range(6)]))), (And([Implies(m < (v["arg4_length"] - 1 + 1), And([Implies(n < (v["arg4_length"] - 1 + 1), Select(v["arg4_values"], m) != Select(v["arg4_values"], n)) for n in range(6)])) for m in range(6)])))) for j in range(6)]))) for i in range(6)]))) if n else
          And(And(And(And(And((0 - v["arg5_ndim"]) <= v["arg1_value"], v["arg1_value"] < v["arg5_ndim"]), (0 - v["arg5_ndim"]) <= v["arg2_value"]), v["arg2_value"] < v["arg5_ndim"]), v["arg3_length"] == v["arg4_length"]), And([Implies(i < (v["arg3_length"] - 1 + 1), And(And((0 - v["arg5_ndim"]) <= Select(v["arg3_values"], i), Select(v["arg3_values"], i) < v["arg5_ndim"]), And([Implies(j < (v["arg4_length"] - 1 + 1), And(And(And((0 - v["arg5_ndim"]) <= Select(v["arg4_values"], j), Select(v["arg4_values"], j) < v["arg5_ndim"]), (And([Implies(k < (v["arg3_length"] - 1 + 1), And([Implies(l < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], k) != Select(v["arg3_values"], l)) for l in range(6)])) for k in range(6)]))), (And([Implies(m < (v["arg4_length"] - 1 + 1), And([Implies(n < (v["arg4_length"] - 1 + 1), Select(v["arg4_values"], m) != Select(v["arg4_values"], n)) for n in range(6)])) for m in range(6)])))) for j in range(6)]))) for i in range(6)])))
)

def rule_53_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 53
        rule_53(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_length': arg4_length, 'arg4_values': arg4_values, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_length': arg4['length'], 'arg4_values': arg4['values'], 'arg5_ndim': arg5['ndim']}, neg)
