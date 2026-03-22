import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# var, accum, linear, grad, lr, l1, l2, lr_power must have at least zero dimensions (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_ndim"] >= 0, v["arg2_ndim"] >= 0), v["arg3_ndim"] >= 0), v["arg4_ndim"] >= 0), v["arg5_ndim"] >= 0), v["arg6_ndim"] >= 0), v["arg7_ndim"] >= 0), v["arg8_ndim"] >= 0)) if n else
          And(And(And(And(And(And(And(v["arg1_ndim"] >= 0, v["arg2_ndim"] >= 0), v["arg3_ndim"] >= 0), v["arg4_ndim"] >= 0), v["arg5_ndim"] >= 0), v["arg6_ndim"] >= 0), v["arg7_ndim"] >= 0), v["arg8_ndim"] >= 0))
)

def rule_15_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_ndim = Int('arg4_ndim')
        arg5_ndim = Int('arg5_ndim')
        arg6_ndim = Int('arg6_ndim')
        arg7_ndim = Int('arg7_ndim')
        arg8_ndim = Int('arg8_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg5_ndim == arg5.ndim)
        solver.add(arg6_ndim == arg6.ndim)
        solver.add(arg7_ndim == arg7.ndim)
        solver.add(arg8_ndim == arg8.ndim)

        # Constraints for rule 15
        rule_15(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_ndim': arg4_ndim, 'arg5_ndim': arg5_ndim, 'arg6_ndim': arg6_ndim, 'arg7_ndim': arg7_ndim, 'arg8_ndim': arg8_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_ndim': arg4['ndim'], 'arg5_ndim': arg5['ndim'], 'arg6_ndim': arg6['ndim'], 'arg7_ndim': arg7['ndim'], 'arg8_ndim': arg8['ndim']}, neg)
