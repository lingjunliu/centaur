import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
rule_51 = lambda s, v, n=False: (
    s.add(Not(
        And(
            v["input_ndim"] >= 0,
            v["input_ndim"] <= MAX_N_DIM,

            # input dimensions positive
            And([Select(v["input_shape"], i) > 0 for i in range(6)]),

            # dtype constraints
            Or([v["input_dtype"] == StringVal(dt) for dt in list_of_available_dtypes]),

            # shift and axis must be integer tensors
            v["shift_dtype"] == StringVal("int32"),
            v["axis_dtype"] == StringVal("int32"),

            # shift and axis same rank
            v["shift_ndim"] == v["axis_ndim"]
        )
    )) if n else
    s.add(
        And(
            v["input_ndim"] >= 0,
            v["input_ndim"] <= MAX_N_DIM,
            And([Select(v["input_shape"], i) > 0 for i in range(6)]),
            Or([v["input_dtype"] == StringVal(dt) for dt in list_of_available_dtypes]),
            v["shift_dtype"] == StringVal("int32"),
            v["axis_dtype"] == StringVal("int32"),
            v["shift_ndim"] == v["axis_ndim"]
        )
    )
)
def rule_51_func(arg1, arg2, arg3, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))
    shift_tensor = next(iter(arg2.values()))
    axis_tensor = next(iter(arg3.values()))

    if not solver:
        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())
        input_dtype = String('input_dtype')

        shift_ndim = Int('shift_ndim')
        shift_dtype = String('shift_dtype')

        axis_ndim = Int('axis_ndim')
        axis_dtype = String('axis_dtype')

        solver.add(input_ndim == input_tensor.ndim)
        solver.add(input_dtype == StringVal(str(input_tensor.dtype)))

        solver.add(shift_ndim == shift_tensor.ndim)
        solver.add(shift_dtype == StringVal(str(shift_tensor.dtype)))

        solver.add(axis_ndim == axis_tensor.ndim)
        solver.add(axis_dtype == StringVal(str(axis_tensor.dtype)))

        for i in range(input_tensor.ndim):
            input_shape = Store(input_shape, i, input_tensor.shape[i])

        rule_51(solver, {
            "input_ndim": input_ndim,
            "input_shape": input_shape,
            "input_dtype": input_dtype,
            "shift_ndim": shift_ndim,
            "shift_dtype": shift_dtype,
            "axis_ndim": axis_ndim,
            "axis_dtype": axis_dtype
        })

        return solver.check() == sat

    else:
        rule_51(solver, {
            "input_ndim": input_tensor["ndim"],
            "input_shape": input_tensor["shape"],
            "input_dtype": input_tensor["dtype"],
            "shift_ndim": shift_tensor["ndim"],
            "shift_dtype": shift_tensor["dtype"],
            "axis_ndim": axis_tensor["ndim"],
            "axis_dtype": axis_tensor["dtype"]
        }, neg)