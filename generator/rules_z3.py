import numpy as np
import logging
from itertools import combinations,permutations

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# hack to circumvent missing definition in import
np.float128 = np.float64

# Log to a file
logging.basicConfig(filename='hacking.log', level=logging.DEBUG)

# Add rule-specific constraints 
_ = lambda s,r,v: {
    "rule_1": lambda s,v: (
        s.add(v["arg1_ndim"] == v["arg2_ndim"]),
        [s.add(Implies(i < v["arg1_ndim"], Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i))) for i in range(MAX_N_DIM)]
    ),
    "rule_2": lambda s,v: (
        s.add(v["arg2"] >= -1 * v["arg1_ndim"]),
        s.add(v["arg2"] <= v["arg1_ndim"] - 1)
    ),
    "rule_4": lambda s,v: (
        s.add(v["arg1_dtype"] == v["arg2_dtype"])
    ),
    "rule_11": lambda s,v: (
        s.add(v["arg3_range"][0] >= 0),
        s.add(v["arg3_range"][1] <= Select(v["arg1_shape"], v["arg2"]) - 1)
    )
}[r](s,v)

'''
    Corresponds to rule that asserts that shapes of tensors are identical (Rule 1)

    Positive Example:
    arg1 = {'a': np.random.rand(3,2)}
    arg2 = {'b': np.random.rand(3,2)}

    Negative Example:
    arg1 = {'a': np.random.rand(3,2,5)}
    arg2 = {'b': np.random.rand(3,2)}
'''

def rule_1_func(arg1, arg2, solver=None):
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))    

    # Invariant learning phase
    if not solver: 
        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        arg1_shape, arg2_shape = Array('arg1_shape', IntSort(), IntSort()), Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1_value.ndim)
        solver.add(arg2_ndim == arg2_value.ndim)
        for i in range(arg1_value.ndim):
            arg1_shape = Store(arg1_shape, i, arg1_value.shape[i])
        for i in range(arg2_value.ndim):
            arg2_shape = Store(arg2_shape, i, arg2_value.shape[i])

        # Constraints for rule 1
        _(solver, 'rule_1', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 
                             'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 1
        _(solver, 'rule_1', {'arg1_ndim': arg1_value['ndim'], 'arg1_shape': arg1_value['shape'], 
                             'arg2_ndim': arg2_value['ndim'], 'arg2_shape': arg2_value['shape']})

'''
    Corresponds to rule that the value in arg2 (dim) is within the range of dimensions of arg1 (input_tensor). (Rule 2)

    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"dim": 1}

    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"dim": 3}   # Invalid dim
'''

def rule_2_func(arg1, arg2, solver=None):
    if next(iter(arg2.keys())) != "dim":
        return False
    
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    
    # Invariant learning phase
    if not solver: 
        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2 = Ints('arg1_ndim arg2')
    
        # Value assignments
        solver.add(arg1_ndim == arg1_value.ndim)
        solver.add(arg2 == int(arg2_value)) 

        # Constraints for rule 2
        _(solver, 'rule_2', {'arg1_ndim': arg1_ndim, 'arg2': arg2})

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 2
        _(solver, 'rule_2', {'arg1_ndim': arg1_value['ndim'], 'arg2': arg2_value})

'''
    Corresponds to rule that asserts that arg1 and arg2 have the same data type. (Rule 4)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.float32)} # float32
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int32)} # int32, not the same as arg1
'''

def rule_4_func(arg1, arg2, solver=None):
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    
    # Invariant learning phase
    if not solver: 
        # Variable declarations
        solver = Solver()
        arg1_dtype, arg2_dtype = Ints('arg1_dtype arg2_dtype')
    
        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1_value.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2_value.dtype))

        # Constraints for rule 4
        _(solver, 'rule_4', {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 4
        _(solver, 'rule_4', {'arg1_dtype': arg1_value['dtype'], 'arg2_dtype': arg2_value['dtype']})

'''
    Corresponds to a rule that ensures the index tensor (arg3) is within
    the constraint of the dimension size of the input tensor (arg1) along
    the specified dim (arg2).
        
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2, 3], [3, 4, 5]])} # Shape: (2, 3)
    arg2 = 1    # dim
    arg3 = {"index": np.array([0, 2])} # Valid index for dim size 3
        
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2, 3], [3, 4, 5]])} # Shape: (2, 3)
    arg2 = 0    # dim
    arg3 = {"index": np.array([0, 2])} # Invalid index for dim size 2
'''

def rule_11_func(arg1, arg2, arg3, solver=None):
    if next(iter(arg2.keys())) != "dim" or next(iter(arg3.keys())) != "index":
        return False

    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    arg3_value = next(iter(arg3.values()))
    
    # Invariant learning phase
    if not solver: 
        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2 = Int('arg2')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
    
        # Value assignments
        for i in range(arg1_value.ndim):
            arg1_shape = Store(arg1_shape, i, arg1_value.shape[i])
        solver.add(arg2 == int(arg2_value))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3_value)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3_value)))

        # Constraints for rule 11
        _(solver, 'rule_11', {'arg1_shape': arg1_shape, 'arg2': arg2, 'arg3_range': arg3_range})

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 11
        _(solver, 'rule_11', {'arg1_shape': arg1_value['shape'], 'arg2': arg2_value, 'arg3_range': arg3_value['range']})


############### mapping ################

rule_func_map = { 
    2: {
        'rule_1': rule_1_func,
        'rule_2': rule_2_func,
        'rule_4': rule_4_func,
    },
    3: {
        'rule_11': rule_11_func
    }
}

# Add rules where the order of arguments does not matter
# i.e. the nature of the arguments are the same
# e.g. two tensors having the same shape: does not matter if the first tensor is arg1 or arg2
# implication: do not check these rules for all permutations of the arguments
order_agnostic_rules = {
    2: ['rule_1', 'rule_4']
}

def check_rules_z3(input_dict, print_rules=False):
    set_of_rules_passed = set()
    
    if len(input_dict.keys()) < 1:
        print("Not enough arguments to check rules")
        return set_of_rules_passed
    
    # Generate unique combinations of keys
    for arity in rule_func_map:
        if len(input_dict.keys()) < arity:
            continue
        for args in combinations(input_dict.keys(), arity):
            for rule_name, z3_func in rule_func_map[arity].items():
                # Check if the rule is satisfied for the current combination of arguments
                # i.e. distance function returns zero
                try:
                    arg_dicts = tuple({k: input_dict[k]} for k in args)
                    if z3_func(*arg_dicts):
                        # If the rule is satisfied, add it to the set of passed rules
                        set_of_rules_passed.add((arity, rule_name, *args))
                except:
                    pass    # The rule is not applicable

                # Change order of arguments to check the rule unless the rule is order-agnostic
                if rule_name not in order_agnostic_rules.get(arity, []):
                    for permuted_args in permutations(args):
                        if permuted_args == args:
                            continue
                        try: 
                            arg_dicts = tuple({k: input_dict[k]} for k in permuted_args)
                            if z3_func(*arg_dicts):
                                # If the rule is satisfied, add it to the set of passed rules
                                set_of_rules_passed.add((arity, rule_name, *permuted_args))
                        except:
                            pass    # The rule is not applicable
    # Optionally print the rules that have been passed
    if print_rules:
        for arity, rule_name, *args in set_of_rules_passed:
            print(f"Arity {arity} Rule {rule_name} passed between {args}")
    # Return the set of rules that have been passed
    return set_of_rules_passed
