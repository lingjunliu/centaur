import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 3: handle must be a scalar resource tensor

rule_3 = lambda s, v, n=False: (
    s.add(Not(
        And(
            v["handle_ndim"] == 0,
            v["handle_dtype"] == StringVal("resource")
        )
    )) if n else
    s.add(
        And(
            v["handle_ndim"] == 0,
            v["handle_dtype"] == StringVal("resource")
        )
    )
)
def rule_3_func(arg1, solver=None, neg=False):

    handle = next(iter(arg1.values()))

    if not solver:
        if not isinstance(handle, np.ndarray):
            return False

        solver = Solver()

        handle_ndim = Int('handle_ndim')
        handle_dtype = String('handle_dtype')

        solver.add(handle_ndim == handle.ndim)
        solver.add(handle_dtype == StringVal(str(handle.dtype)))

        rule_3(solver, {
            "handle_ndim": handle_ndim,
            "handle_dtype": handle_dtype
        })

        return solver.check() == sat

    else:
        rule_3(solver, {
            "handle_ndim": handle["ndim"],
            "handle_dtype": handle["dtype"]
        }, neg)