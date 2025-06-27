import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# The value tensor v_1 must be representable in Half dtype without overflow (Rule 997)

rule_997 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_range"], 0) > -65504, Select(v["arg1_range"], 1) < 65504)) if n else
          And(Select(v["arg1_range"], 0) > -65504, Select(v["arg1_range"], 1) < 65504))
)

def rule_997_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 997
        rule_997(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_997(solver, {'arg1_range': arg1['range']}, neg)
