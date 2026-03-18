import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 1: elem must be a valid tensor when swap_memory is default (False)

rule_1 = lambda s, v, n=False: (
    s.add(Not(
        And(
            v["elem_ndim"] >= 0,
            v["elem_ndim"] <= MAX_N_DIM,
            And([Select(v["elem_shape"], i) > 0 for i in range(6)]),
            Or([v["elem_dtype"] == StringVal(dt) for dt in list_of_available_dtypes])
        )
    )) if n else
    s.add(
        And(
            v["elem_ndim"] >= 0,
            v["elem_ndim"] <= MAX_N_DIM,
            And([Select(v["elem_shape"], i) > 0 for i in range(6)]),
            Or([v["elem_dtype"] == StringVal(dt) for dt in list_of_available_dtypes])
        )
    )
)
def rule_1_func(arg1, solver=None, neg=False):

    elem = next(iter(arg1.values()))

    if not solver:
        if not isinstance(elem, np.ndarray):
            return False

        solver = Solver()

        elem_ndim = Int('elem_ndim')
        elem_shape = Array('elem_shape', IntSort(), IntSort())
        elem_dtype = String('elem_dtype')

        solver.add(elem_ndim == elem.ndim)
        solver.add(elem_dtype == StringVal(str(elem.dtype)))

        for i in range(elem.ndim):
            elem_shape = Store(elem_shape, i, elem.shape[i])

        rule_1(solver, {
            "elem_ndim": elem_ndim,
            "elem_shape": elem_shape,
            "elem_dtype": elem_dtype
        })

        return solver.check() == sat

    else:
        rule_1(solver, {
            "elem_ndim": elem['ndim'],
            "elem_shape": elem['shape'],
            "elem_dtype": elem['dtype']
        }, neg)