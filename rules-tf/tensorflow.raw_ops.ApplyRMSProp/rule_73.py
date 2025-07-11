import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, ms and mom must have the same shape and dtype and all input tensors must have floating point dtypes (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] == v["arg2_ndim"], v["arg1_ndim"] == v["arg3_ndim"]), v["arg1_ndim"] == v["arg4_ndim"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8)), Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8)), Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8)), Or(v["arg6_dtype"] == 7, v["arg6_dtype"] == 8)), Or(v["arg7_dtype"] == 7, v["arg7_dtype"] == 8)), Or(v["arg8_dtype"] == 7, v["arg8_dtype"] == 8))) for i in range(6)]))) if n else
          And(And(And(v["arg1_ndim"] == v["arg2_ndim"], v["arg1_ndim"] == v["arg3_ndim"]), v["arg1_ndim"] == v["arg4_ndim"]), And([Implies(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]), Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8)), Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8)), Or(v["arg5_dtype"] == 7, v["arg5_dtype"] == 8)), Or(v["arg6_dtype"] == 7, v["arg6_dtype"] == 8)), Or(v["arg7_dtype"] == 7, v["arg7_dtype"] == 8)), Or(v["arg8_dtype"] == 7, v["arg8_dtype"] == 8))) for i in range(6)])))
)

def rule_73_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')
        arg6_dtype = Int('arg6_dtype')
        arg7_dtype = Int('arg7_dtype')
        arg8_dtype = Int('arg8_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))
        solver.add(arg6_dtype == list_of_available_dtypes.index(arg6.dtype))
        solver.add(arg7_dtype == list_of_available_dtypes.index(arg7.dtype))
        solver.add(arg8_dtype == list_of_available_dtypes.index(arg8.dtype))

        # Constraints for rule 73
        rule_73(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg4_shape': arg4_shape, 'arg4_dtype': arg4_dtype, 'arg4_ndim': arg4_ndim, 'arg5_dtype': arg5_dtype, 'arg6_dtype': arg6_dtype, 'arg7_dtype': arg7_dtype, 'arg8_dtype': arg8_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg4_shape': arg4['shape'], 'arg4_dtype': arg4['dtype'], 'arg4_ndim': arg4['ndim'], 'arg5_dtype': arg5['dtype'], 'arg6_dtype': arg6['dtype'], 'arg7_dtype': arg7['dtype'], 'arg8_dtype': arg8['dtype']}, neg)
