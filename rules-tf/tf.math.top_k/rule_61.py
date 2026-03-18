import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 61: valid k provided

rule_61 = lambda s, v, n=False: (
    s.add(Not(And(

        v["arg1_ndim"] >= 1,

        Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0,

        # k constraints
        v["k"] >= 1,
        v["k"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1),

        v["arg1_dtype"] != StringVal("bool")

    )) if n else
    And(

        v["arg1_ndim"] >= 1,

        Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0,

        v["k"] >= 1,
        v["k"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1),

        v["arg1_dtype"] != StringVal("bool")
    ))
)

def rule_61_func(arg1, arg2, solver=None, neg=False):

    arg1 = next(iter(arg1.values()))
    k = next(iter(arg2.values()))

    if not solver:

        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(k, int):
            return False

        solver = Solver()

        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = String('arg1_dtype')
        k_sym = Int('k')

        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == StringVal(str(arg1.dtype)))
        solver.add(k_sym == k)

        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        rule_61(
            solver,
            {
                "arg1_ndim": arg1_ndim,
                "arg1_shape": arg1_shape,
                "arg1_dtype": arg1_dtype,
                "k": k_sym
            }
        )

        return solver.check() == sat

    else:

        rule_61(
            solver,
            {
                "arg1_ndim": arg1["ndim"],
                "arg1_shape": arg1["shape"],
                "arg1_dtype": arg1["dtype"],
                "k": k
            },
            neg
        )