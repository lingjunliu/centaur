import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a tensor's shape at dimension 0 is greater than 0, then its minimum value must be an integer (Rule 269)

rule_269 = lambda s, v: (
    s.add(If(Select(v["arg1_shape"], 0) > 0, And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), False))
)

def rule_269_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 269
        rule_269(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_269(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']})
