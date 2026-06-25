import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the ksize is [1,1,1,1,1] and strides is [1,1,1,1,1] and the padding is VALID then the orig_input, orig_output and grad should have same shape (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(And(And((And(And(And(And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 1) == 1), Select(v["arg1_values"], 2) == 1), Select(v["arg1_values"], 3) == 1), Select(v["arg1_values"], 4) == 1)), (And(And(And(And(Select(v["arg2_values"], 0) == 1, Select(v["arg2_values"], 1) == 1), Select(v["arg2_values"], 2) == 1), Select(v["arg2_values"], 3) == 1), Select(v["arg2_values"], 4) == 1))), (v["arg6_value"] == 28)), And(v["arg3_ndim"] == v["arg4_ndim"], And([Implies(i < (v["arg3_ndim"] - 1 + 1), And(And(Select(v["arg3_shape"], i) == Select(v["arg4_shape"], i), v["arg3_ndim"] == v["arg5_ndim"]), And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == Select(v["arg5_shape"], i)) for i in range(6)]))) for i in range(6)])), True)) if n else
          If(And(And((And(And(And(And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 1) == 1), Select(v["arg1_values"], 2) == 1), Select(v["arg1_values"], 3) == 1), Select(v["arg1_values"], 4) == 1)), (And(And(And(And(Select(v["arg2_values"], 0) == 1, Select(v["arg2_values"], 1) == 1), Select(v["arg2_values"], 2) == 1), Select(v["arg2_values"], 3) == 1), Select(v["arg2_values"], 4) == 1))), (v["arg6_value"] == 28)), And(v["arg3_ndim"] == v["arg4_ndim"], And([Implies(i < (v["arg3_ndim"] - 1 + 1), And(And(Select(v["arg3_shape"], i) == Select(v["arg4_shape"], i), v["arg3_ndim"] == v["arg5_ndim"]), And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == Select(v["arg5_shape"], i)) for i in range(6)]))) for i in range(6)])), True))
)

def rule_45_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())
        arg6_value = Int('arg6_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_ndim == arg5.ndim)
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])
        solver.add(arg6_value == list_of_string_values_tf.index(arg6))

        # Constraints for rule 45
        rule_45(solver, {'arg1_values': arg1_values, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim, 'arg5_shape': arg5_shape, 'arg5_ndim': arg5_ndim, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_values': arg1['values'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim'], 'arg5_shape': arg5['shape'], 'arg5_ndim': arg5['ndim'], 'arg6_value': arg6['value']}, neg)
