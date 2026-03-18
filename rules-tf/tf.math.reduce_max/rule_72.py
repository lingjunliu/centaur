import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 72: axis is a valid list of integers

rule_72 = lambda s, v, n=False: (
    s.add(Not(And(

        v["arg1_ndim"] >= 1,

        # tensor must be non-empty
        And([Implies(i < v["arg1_ndim"],
                     Select(v["arg1_shape"], i) > 0)
             for i in range(MAX_N_DIM)]),

        # dtype numeric
        v["arg1_dtype"] != StringVal("bool"),

        # axis length constraint
        v["axis_len"] >= 1,
        v["axis_len"] <= v["arg1_ndim"],

        # each axis valid
        And([Implies(i < v["axis_len"],
                     And(v["axis"][i] >= -v["arg1_ndim"],
                         v["axis"][i] < v["arg1_ndim"]))
             for i in range(MAX_N_DIM)]),

        # no duplicate axes
        Distinct([v["axis"][i] for i in range(MAX_N_DIM)])
    )) if n else
    And(
        v["arg1_ndim"] >= 1,

        And([Implies(i < v["arg1_ndim"],
                     Select(v["arg1_shape"], i) > 0)
             for i in range(MAX_N_DIM)]),

        v["arg1_dtype"] != StringVal("bool"),

        v["axis_len"] >= 1,
        v["axis_len"] <= v["arg1_ndim"],

        And([Implies(i < v["axis_len"],
                     And(v["axis"][i] >= -v["arg1_ndim"],
                         v["axis"][i] < v["arg1_ndim"]))
             for i in range(MAX_N_DIM)]),

        Distinct([v["axis"][i] for i in range(MAX_N_DIM)])
    ))
)
def rule_72_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(axis, (list, tuple)):
            return False

        solver = Solver()

        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = String('arg1_dtype')

        axis_len = Int('axis_len')
        axis_arr = Array('axis', IntSort(), IntSort())

        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == StringVal(str(arg1.dtype)))
        solver.add(axis_len == len(axis))

        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        for i in range(len(axis)):
            axis_arr = Store(axis_arr, i, axis[i])

        rule_72(solver, {
            "arg1_ndim": arg1_ndim,
            "arg1_shape": arg1_shape,
            "arg1_dtype": arg1_dtype,
            "axis_len": axis_len,
            "axis": axis_arr
        })

        return solver.check() == sat

    # Fuzz generation phase
    else:
        rule_72(solver, {
            "arg1_ndim": arg1["ndim"],
            "arg1_shape": arg1["shape"],
            "arg1_dtype": arg1["dtype"],
            "axis_len": axis["len"],
            "axis": axis["values"]
        }, neg)