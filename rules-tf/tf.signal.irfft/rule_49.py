import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 49: fft_length is provided
# last_dim == fft_length // 2 + 1

rule_49 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["input_ndim"] >= 1,
                v["fft_length_ndim"] == 1,
                Select(v["fft_length_shape"], 0) == 1,
                Select(v["fft_length"], 0) > 0,
                Select(v["input_shape"], v["input_ndim"] - 1) ==
                (Select(v["fft_length"], 0) / 2) + 1,
                And([Implies(i < v["input_ndim"],
                             Select(v["input_shape"], i) > 0)
                     for i in range(MAX_N_DIM)])
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 1,
            v["fft_length_ndim"] == 1,
            Select(v["fft_length_shape"], 0) == 1,
            Select(v["fft_length"], 0) > 0,
            Select(v["input_shape"], v["input_ndim"] - 1) ==
            (Select(v["fft_length"], 0) / 2) + 1,
            And([Implies(i < v["input_ndim"],
                         Select(v["input_shape"], i) > 0)
                 for i in range(MAX_N_DIM)])
        )
    )
)
def rule_49_func(arg1, arg2, solver=None, neg=False):
    input_tensor = next(iter(arg1.values()))
    fft_length = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(input_tensor, np.ndarray):
            return False
        if not isinstance(fft_length, np.ndarray):
            return False
        if fft_length.size != 1:
            return False

        solver = Solver()

        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())

        fft_length_ndim = Int('fft_length_ndim')
        fft_length_shape = Array('fft_length_shape', IntSort(), IntSort())
        fft_length_val = Array('fft_length', IntSort(), IntSort())

        solver.add(input_ndim == input_tensor.ndim)
        for i in range(input_tensor.ndim):
            solver.add(Select(input_shape, i) == input_tensor.shape[i])

        solver.add(fft_length_ndim == fft_length.ndim)
        for i in range(fft_length.ndim):
            solver.add(Select(fft_length_shape, i) == fft_length.shape[i])

        solver.add(Select(fft_length_val, 0) == int(fft_length.flatten()[0]))

        rule_49(solver, {
            "input_ndim": input_ndim,
            "input_shape": input_shape,
            "fft_length_ndim": fft_length_ndim,
            "fft_length_shape": fft_length_shape,
            "fft_length": fft_length_val
        })

        return solver.check() == sat

    # Fuzz generation phase
    else:
        rule_49(
            solver,
            {
                "input_ndim": input_tensor["ndim"],
                "input_shape": input_tensor["shape"],
                "fft_length_ndim": fft_length["ndim"],
                "fft_length_shape": fft_length["shape"],
                "fft_length": fft_length["value"]
            },
            neg
        )