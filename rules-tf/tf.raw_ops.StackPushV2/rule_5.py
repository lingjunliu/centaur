import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 5: elem must match stack dtype and shape when handle is valid

rule_5 = lambda s, v, n=False: (
    s.add(Not(
        And(
            v["elem_dtype"] == v["stack_dtype"],
            v["elem_ndim"] == v["stack_ndim"],
            And([Select(v["elem_shape"], i) == Select(v["stack_shape"], i) for i in range(6)])
        )
    )) if n else
    s.add(
        And(
            v["elem_dtype"] == v["stack_dtype"],
            v["elem_ndim"] == v["stack_ndim"],
            And([Select(v["elem_shape"], i) == Select(v["stack_shape"], i) for i in range(6)])
        )
    )
)
def rule_5_func(arg1, arg2, solver=None, neg=False):

    elem = next(iter(arg1.values()))
    stack_spec = next(iter(arg2.values()))

    if not solver:
        if not isinstance(elem, np.ndarray):
            return False

        solver = Solver()

        elem_ndim = Int('elem_ndim')
        elem_shape = Array('elem_shape', IntSort(), IntSort())
        elem_dtype = String('elem_dtype')

        stack_ndim = Int('stack_ndim')
        stack_shape = Array('stack_shape', IntSort(), IntSort())
        stack_dtype = String('stack_dtype')

        # Assign actual values
        solver.add(elem_ndim == elem.ndim)
        solver.add(elem_dtype == StringVal(str(elem.dtype)))

        solver.add(stack_ndim == stack_spec["ndim"])
        solver.add(stack_dtype == StringVal(stack_spec["dtype"]))

        for i in range(elem.ndim):
            elem_shape = Store(elem_shape, i, elem.shape[i])

        for i in range(stack_spec["ndim"]):
            stack_shape = Store(stack_shape, i, stack_spec["shape"][i])

        rule_5(solver, {
            "elem_ndim": elem_ndim,
            "elem_shape": elem_shape,
            "elem_dtype": elem_dtype,
            "stack_ndim": stack_ndim,
            "stack_shape": stack_shape,
            "stack_dtype": stack_dtype
        })

        return solver.check() == sat

    else:
        rule_5(solver, {
            "elem_ndim": elem["ndim"],
            "elem_shape": elem["shape"],
            "elem_dtype": elem["dtype"],
            "stack_ndim": stack_spec["ndim"],
            "stack_shape": stack_spec["shape"],
            "stack_dtype": stack_spec["dtype"]
        }, neg)