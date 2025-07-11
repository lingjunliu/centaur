import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If m, v, beta and gamma are not provided, t_min t_max m_min m_max v_min v_max beta_min beta_max gamma_min and gamma_max type must be Tensor (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg11_ndim"] != 1, v["arg12_ndim"] != 1), v["arg13_ndim"] != 1), v["arg14_ndim"] != 1), And(And(And(And(And(And(And(And(And(v["arg1_ndim"] >= 0, v["arg2_ndim"] >= 0), v["arg3_ndim"] >= 0), v["arg4_ndim"] >= 0), v["arg5_ndim"] >= 0), v["arg6_ndim"] >= 0), v["arg7_ndim"] >= 0), v["arg8_ndim"] >= 0), v["arg9_ndim"] >= 0), v["arg10_ndim"] >= 0), False)) if n else
          If(Or(Or(Or(v["arg11_ndim"] != 1, v["arg12_ndim"] != 1), v["arg13_ndim"] != 1), v["arg14_ndim"] != 1), And(And(And(And(And(And(And(And(And(v["arg1_ndim"] >= 0, v["arg2_ndim"] >= 0), v["arg3_ndim"] >= 0), v["arg4_ndim"] >= 0), v["arg5_ndim"] >= 0), v["arg6_ndim"] >= 0), v["arg7_ndim"] >= 0), v["arg8_ndim"] >= 0), v["arg9_ndim"] >= 0), v["arg10_ndim"] >= 0), False))
)

def rule_50_func(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))
    arg7 = next(iter(arg7.values()))
    arg8 = next(iter(arg8.values()))
    arg9 = next(iter(arg9.values()))
    arg10 = next(iter(arg10.values()))
    arg11 = next(iter(arg11.values()))
    arg12 = next(iter(arg12.values()))
    arg13 = next(iter(arg13.values()))
    arg14 = next(iter(arg14.values()))

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
        if not isinstance(arg9, np.ndarray):
            return False
        if not isinstance(arg10, np.ndarray):
            return False
        if not isinstance(arg11, np.ndarray):
            return False
        if not isinstance(arg12, np.ndarray):
            return False
        if not isinstance(arg13, np.ndarray):
            return False
        if not isinstance(arg14, np.ndarray):
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
        arg9_ndim = Int('arg9_ndim')
        arg10_ndim = Int('arg10_ndim')
        arg11_ndim = Int('arg11_ndim')
        arg12_ndim = Int('arg12_ndim')
        arg13_ndim = Int('arg13_ndim')
        arg14_ndim = Int('arg14_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg5_ndim == arg5.ndim)
        solver.add(arg6_ndim == arg6.ndim)
        solver.add(arg7_ndim == arg7.ndim)
        solver.add(arg8_ndim == arg8.ndim)
        solver.add(arg9_ndim == arg9.ndim)
        solver.add(arg10_ndim == arg10.ndim)
        solver.add(arg11_ndim == arg11.ndim)
        solver.add(arg12_ndim == arg12.ndim)
        solver.add(arg13_ndim == arg13.ndim)
        solver.add(arg14_ndim == arg14.ndim)

        # Constraints for rule 50
        rule_50(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_ndim': arg4_ndim, 'arg5_ndim': arg5_ndim, 'arg6_ndim': arg6_ndim, 'arg7_ndim': arg7_ndim, 'arg8_ndim': arg8_ndim, 'arg9_ndim': arg9_ndim, 'arg10_ndim': arg10_ndim, 'arg11_ndim': arg11_ndim, 'arg12_ndim': arg12_ndim, 'arg13_ndim': arg13_ndim, 'arg14_ndim': arg14_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_ndim': arg4['ndim'], 'arg5_ndim': arg5['ndim'], 'arg6_ndim': arg6['ndim'], 'arg7_ndim': arg7['ndim'], 'arg8_ndim': arg8['ndim'], 'arg9_ndim': arg9['ndim'], 'arg10_ndim': arg10['ndim'], 'arg11_ndim': arg11['ndim'], 'arg12_ndim': arg12['ndim'], 'arg13_ndim': arg13['ndim'], 'arg14_ndim': arg14['ndim']}, neg)
