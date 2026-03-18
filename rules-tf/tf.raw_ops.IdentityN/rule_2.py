import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: IdentityN input constraints

rule_2 = lambda s, v, n=False: (
    s.add(Not(And(

        # at least one tensor
        v["num_inputs"] >= 1,

        # valid rank for each tensor
        And([
            Implies(i < v["num_inputs"],
                    And(v["inputs_ndim"][i] >= 0,
                        v["inputs_ndim"][i] <= MAX_N_DIM))
            for i in range(MAX_SZ_NUM)
        ]),

        # valid dimension sizes
        And([
            Implies(
                And(i < v["num_inputs"],
                    j < v["inputs_ndim"][i]),
                Select(v["inputs_shape"][i], j) > 0
            )
            for i in range(MAX_SZ_NUM)
            for j in range(MAX_N_DIM)
        ]),

        # dtype must be valid TF dtype
        And([
            Implies(
                i < v["num_inputs"],
                Or([v["inputs_dtype"][i] == StringVal(dt)
                    for dt in list_of_available_dtypes])
            )
            for i in range(MAX_SZ_NUM)
        ])

    ))) if n else
    And(

        v["num_inputs"] >= 1,

        And([
            Implies(i < v["num_inputs"],
                    And(v["inputs_ndim"][i] >= 0,
                        v["inputs_ndim"][i] <= MAX_N_DIM))
            for i in range(MAX_SZ_NUM)
        ]),

        And([
            Implies(
                And(i < v["num_inputs"],
                    j < v["inputs_ndim"][i]),
                Select(v["inputs_shape"][i], j) > 0
            )
            for i in range(MAX_SZ_NUM)
            for j in range(MAX_N_DIM)
        ]),

        And([
            Implies(
                i < v["num_inputs"],
                Or([v["inputs_dtype"][i] == StringVal(dt)
                    for dt in list_of_available_dtypes])
            )
            for i in range(MAX_SZ_NUM)
        ])
    )
)

def rule_2_func(arg1, solver=None, neg=False):

    inputs = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:

        if not isinstance(inputs, (list, tuple)):
            return False
        if len(inputs) == 0:
            return False
        if not all(isinstance(t, np.ndarray) for t in inputs):
            return False

        solver = Solver()

        num_inputs = Int('num_inputs')
        inputs_ndim = Array('inputs_ndim', IntSort(), IntSort())
        inputs_shape = Array('inputs_shape', IntSort(), ArraySort(IntSort(), IntSort()))
        inputs_dtype = Array('inputs_dtype', IntSort(), StringSort())

        solver.add(num_inputs == len(inputs))

        for i, t in enumerate(inputs):

            inputs_ndim = Store(inputs_ndim, i, t.ndim)
            inputs_dtype = Store(inputs_dtype, i, StringVal(str(t.dtype)))

            shape_arr = Array(f'shape_{i}', IntSort(), IntSort())
            for j in range(t.ndim):
                shape_arr = Store(shape_arr, j, t.shape[j])

            inputs_shape = Store(inputs_shape, i, shape_arr)

        rule_2(
            solver,
            {
                "num_inputs": num_inputs,
                "inputs_ndim": inputs_ndim,
                "inputs_shape": inputs_shape,
                "inputs_dtype": inputs_dtype
            }
        )

        return solver.check() == sat

    # Fuzz generation phase
    else:

        rule_2(
            solver,
            {
                "num_inputs": inputs["num_inputs"],
                "inputs_ndim": inputs["ndim"],
                "inputs_shape": inputs["shape"],
                "inputs_dtype": inputs["dtype"]
            },
            neg
        )