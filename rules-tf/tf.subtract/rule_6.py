import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

def broadcastable(x_ndim, x_shape, y_ndim, y_shape):
    constraints = []
    for i in range(MAX_N_DIM):
        xi = If(i < x_ndim, Select(x_shape, x_ndim - 1 - i), 1)
        yi = If(i < y_ndim, Select(y_shape, y_ndim - 1 - i), 1)

        constraints.append(
            Or(
                xi == yi,
                xi == 1,
                yi == 1
            )
        )
    return And(constraints)

rule_6 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["y_ndim"] >= 0,
                v["y_dtype_valid"],
                broadcastable(
                    v["x_ndim"], v["x_shape"],
                    v["y_ndim"], v["y_shape"]
                )
            )
        )
    ) if n else
    s.add(
        And(
            v["y_ndim"] >= 0,
            v["y_dtype_valid"],
            broadcastable(
                v["x_ndim"], v["x_shape"],
                v["y_ndim"], v["y_shape"]
            )
        )
    )
)
def rule_6_func(arg1, arg2, solver=None, neg=False):
    x = next(iter(arg1.values()))
    y = next(iter(arg2.values()))

    if not solver:
        if not isinstance(y, np.ndarray):
            return False

        numeric_types = [
            np.int8, np.int16, np.int32, np.int64,
            np.float16, np.float32, np.float64,
            np.complex64, np.complex128
        ]

        if y.dtype.type not in numeric_types:
            return False

        try:
            np.broadcast_shapes(x.shape, y.shape)
        except:
            return False

        return True

    else:
        rule_6(
            solver,
            {
                "x_ndim": x["ndim"],
                "x_shape": x["shape"],
                "y_ndim": y["ndim"],
                "y_shape": y["shape"],
                "y_dtype_valid": y["dtype_valid"]
            },
            neg
        )