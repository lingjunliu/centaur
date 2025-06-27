import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if tanh is chosen, the product of shapes should be greater than zero  (Rule 147)

rule_147 = lambda s, v, n=False: (
    s.add(Not(If((v["arg2_value"] == 11), Or([And(i < (v["arg1_ndim"] - 1 + 1), Or([And(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) * Select(v["arg1_shape"], j) > 0) for j in range(6)])) for i in range(6)]), False)) if n else
          If((v["arg2_value"] == 11), Or([And(i < (v["arg1_ndim"] - 1 + 1), Or([And(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) * Select(v["arg1_shape"], j) > 0) for j in range(6)])) for i in range(6)]), False))
)

def rule_147_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 147
        rule_147(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_147(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
