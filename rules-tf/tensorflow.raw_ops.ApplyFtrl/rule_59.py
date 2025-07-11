import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, accum, linear, grad, lr, l1, l2 and lr_power must have compatible dtypes and shapes (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_dtype"] == v["arg2_dtype"], v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), v["arg5_dtype"] == v["arg1_dtype"]), v["arg6_dtype"] == v["arg1_dtype"]), v["arg7_dtype"] == v["arg1_dtype"]), v["arg8_dtype"] == v["arg1_dtype"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i), v["arg5_ndim"] == 0), v["arg6_ndim"] == 0), v["arg7_ndim"] == 0), v["arg8_ndim"] == 0)) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))) if n else
          And(And(And(And(And(And(And(v["arg1_dtype"] == v["arg2_dtype"], v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), v["arg5_dtype"] == v["arg1_dtype"]), v["arg6_dtype"] == v["arg1_dtype"]), v["arg7_dtype"] == v["arg1_dtype"]), v["arg8_dtype"] == v["arg1_dtype"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i), v["arg5_ndim"] == 0), v["arg6_ndim"] == 0), v["arg7_ndim"] == 0), v["arg8_ndim"] == 0)) for i in range(6)]))) for i in range(6)]))) for i in range(6)])))
)

def rule_59_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False
        if not isinstance(arg6, np.ndarray):
            return False
        if not isinstance(arg7, np.ndarray):
            return False
        if not isinstance(arg8, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')
        arg5_ndim = Int('arg5_ndim')
        arg5_dtype = Int('arg5_dtype')
        arg6_ndim = Int('arg6_ndim')
        arg6_dtype = Int('arg6_dtype')
        arg7_ndim = Int('arg7_ndim')
        arg7_dtype = Int('arg7_dtype')
        arg8_ndim = Int('arg8_ndim')
        arg8_dtype = Int('arg8_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_ndim == arg5.ndim)
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))
        solver.add(arg6_ndim == arg6.ndim)
        solver.add(arg6_dtype == list_of_available_dtypes.index(arg6.dtype))
        solver.add(arg7_ndim == arg7.ndim)
        solver.add(arg7_dtype == list_of_available_dtypes.index(arg7.dtype))
        solver.add(arg8_ndim == arg8.ndim)
        solver.add(arg8_dtype == list_of_available_dtypes.index(arg8.dtype))

        # Constraints for rule 59
        rule_59(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg4_dtype': arg4_dtype, 'arg4_shape': arg4_shape, 'arg5_dtype': arg5_dtype, 'arg5_ndim': arg5_ndim, 'arg6_dtype': arg6_dtype, 'arg6_ndim': arg6_ndim, 'arg7_dtype': arg7_dtype, 'arg7_ndim': arg7_ndim, 'arg8_dtype': arg8_dtype, 'arg8_ndim': arg8_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg4_dtype': arg4['dtype'], 'arg4_shape': arg4['shape'], 'arg5_dtype': arg5['dtype'], 'arg5_ndim': arg5['ndim'], 'arg6_dtype': arg6['dtype'], 'arg6_ndim': arg6['ndim'], 'arg7_dtype': arg7['dtype'], 'arg7_ndim': arg7['ndim'], 'arg8_dtype': arg8['dtype'], 'arg8_ndim': arg8['ndim']}, neg)
