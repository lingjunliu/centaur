import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 128: valid axis provided

rule_128 = lambda s, v, n=False: (
    s.add(Not(And(

        v["arg1_ndim"] >= 1,

        # valid tensor shape
        And([
            Implies(i < v["arg1_ndim"],
                    Select(v["arg1_shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ]),

        # axis validity
        v["axis"] >= -v["arg1_ndim"],
        v["axis"] < v["arg1_ndim"],

        # floating dtype constraint
        v["arg1_dtype"] != StringVal("int32"),
        v["arg1_dtype"] != StringVal("int64"),
        v["arg1_dtype"] != StringVal("bool")

    )) if n else
    And(

        v["arg1_ndim"] >= 1,

        And([
            Implies(i < v["arg1_ndim"],
                    Select(v["arg1_shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ]),

        v["axis"] >= -v["arg1_ndim"],
        v["axis"] < v["arg1_ndim"],

        v["arg1_dtype"] != StringVal("int32"),
        v["arg1_dtype"] != StringVal("int64"),
        v["arg1_dtype"] != StringVal("bool")
    ))
)
def rule_128_func(arg1, arg2, solver=None, neg=False):

    arg1 = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))

    if not solver:

        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(axis, int):
            return False

        solver = Solver()

        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = String('arg1_dtype')
        axis_sym = Int('axis')

        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == StringVal(str(arg1.dtype)))
        solver.add(axis_sym == axis)

        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        rule_128(
            solver,
            {
                "arg1_ndim": arg1_ndim,
                "arg1_shape": arg1_shape,
                "arg1_dtype": arg1_dtype,
                "axis": axis_sym
            }
        )

        return solver.check() == sat

    else:

        rule_128(
            solver,
            {
                "arg1_ndim": arg1["ndim"],
                "arg1_shape": arg1["shape"],
                "arg1_dtype": arg1["dtype"],
                "axis": axis
            },
            neg
        )