import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 1: General broadcasting rule for tf.broadcast_to

rule_1 = lambda s, v, n=False: (
    s.add(Not(
        And(
            # Rank constraint
            v["arg1_ndim"] <= v["arg2_ndim"],

            # Dimension compatibility
            And([
                Or(
                    # If input dimension exists
                    And(
                        v["arg1_ndim"] - 1 - i >= 0,
                        Or(
                            Select(v["arg1_shape"], v["arg1_ndim"] - 1 - i) ==
                            Select(v["arg2_shape"], v["arg2_ndim"] - 1 - i),

                            Select(v["arg1_shape"], v["arg1_ndim"] - 1 - i) == 1
                        )
                    ),

                    # If input dimension does not exist (implicitly 1)
                    v["arg1_ndim"] - 1 - i < 0
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )) if n else
    s.add(
        And(
            v["arg1_ndim"] <= v["arg2_ndim"],

            And([
                Or(
                    And(
                        v["arg1_ndim"] - 1 - i >= 0,
                        Or(
                            Select(v["arg1_shape"], v["arg1_ndim"] - 1 - i) ==
                            Select(v["arg2_shape"], v["arg2_ndim"] - 1 - i),

                            Select(v["arg1_shape"], v["arg1_ndim"] - 1 - i) == 1
                        )
                    ),
                    v["arg1_ndim"] - 1 - i < 0
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_1_func(arg1, arg2, solver=None, neg=False):
    """
    arg_dict contains:
        {
            "input": {"ndim": ..., "shape": ...},
            "shape": {"ndim": ..., "shape": ...}
        }
    """

    input_tensor = next(iter(arg1.values()))
    target_tensor = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        # Declare symbolic variables
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Assign ranks
        solver.add(arg1_ndim == input_tensor.ndim)
        solver.add(arg2_ndim == len(target_tensor))

        # Assign shapes
        for i in range(input_tensor.ndim):
            solver.add(Select(arg1_shape, i) == input_tensor.shape[i])

        for i in range(len(target_tensor)):
            solver.add(Select(arg2_shape, i) == target_tensor[i])

        # Apply broadcasting rule
        rule_1(
            solver,
            {
                "arg1_ndim": arg1_ndim,
                "arg1_shape": arg1_shape,
                "arg2_ndim": arg2_ndim,
                "arg2_shape": arg2_shape,
            }
        )

        return solver.check() == sat

    # Fuzzing / symbolic generation phase
    else:
        rule_1(
            solver,
            {
                "arg1_ndim": input_tensor["ndim"],
                "arg1_shape": input_tensor["shape"],
                "arg2_ndim": target_tensor["ndim"],
                "arg2_shape": target_tensor["shape"],
            },
            neg
        )