import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 3: MatrixInverse (adjoint=False)

rule_3 = lambda s, v, n=False: (
    s.add(Not(And(

        # rank >= 2
        v["ndim"] >= 2,

        # valid rank bound
        v["ndim"] <= MAX_N_DIM,

        # square matrix condition
        Select(v["shape"], v["ndim"] - 1) ==
        Select(v["shape"], v["ndim"] - 2),

        # positive dimensions
        And([
            Implies(i < v["ndim"],
                    Select(v["shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ]),

        # valid dtype (float/complex only)
        Or([v["dtype"] == StringVal(dt)
            for dt in ["float16","float32","float64",
                       "complex64","complex128"]])

    ))) if n else
    And(

        v["ndim"] >= 2,
        v["ndim"] <= MAX_N_DIM,

        Select(v["shape"], v["ndim"] - 1) ==
        Select(v["shape"], v["ndim"] - 2),

        And([
            Implies(i < v["ndim"],
                    Select(v["shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ]),

        Or([v["dtype"] == StringVal(dt)
            for dt in ["float16","float32","float64",
                       "complex64","complex128"]])
    )
)
def rule_3_func(arg1, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:

        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        ndim = Int('ndim')
        shape = Array('shape', IntSort(), IntSort())
        dtype = String('dtype')

        solver.add(ndim == input_tensor.ndim)
        solver.add(dtype == StringVal(str(input_tensor.dtype)))

        for i in range(input_tensor.ndim):
            shape = Store(shape, i, input_tensor.shape[i])

        rule_3(solver, {
            "ndim": ndim,
            "shape": shape,
            "dtype": dtype
        })

        return solver.check() == sat

    # Fuzz generation phase
    else:
        rule_3(
            solver,
            {
                "ndim": input_tensor["ndim"],
                "shape": input_tensor["shape"],
                "dtype": input_tensor["dtype"]
            },
            neg
        )