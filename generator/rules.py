import numpy as np
import logging
from itertools import combinations,permutations

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes

# hack to circumvent missing definition in import
np.float128 = np.float64

# Log to a file
logging.basicConfig(filename='hacking.log', level=logging.DEBUG)

'''
    Corresponds to rule that asserts that shapes of tensors are identical (Rule 1)

    Positive Example:
    arg1 = {'a': np.random.rand(3,2)}
    arg2 = {'b': np.random.rand(3,2)}

    Negative Example:
    arg1 = {'a': np.random.rand(3,2,5)}
    arg2 = {'b': np.random.rand(3,2)}
'''
def dist_1_rev(arg1, arg2):
    w1 = 0.7
    w2 = 0.3
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))    
    # components of the objective function
    component_MAX_N_DIM = w1*abs(arg1_value.ndim-arg2_value.ndim)/MAX_N_DIM
    l1 = arg1_value.shape
    l2 = arg2_value.shape
    common_part = min(len(l1), len(l2))
    elementwise_distance = [ abs(l1[i]-l2[i]) for i in range(common_part) ]
    component_elements = w2*np.sum(elementwise_distance)/(common_part*MAX_SZ_DIM)
    return component_MAX_N_DIM + component_elements

'''
    Corresponds to rule that the value in arg2 (dim) is within the range of dimensions of arg1 (input_tensor). (Rule 2)

    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"dim": 1}

    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"dim": 3}   # Invalid dim
'''
def dist_2_rev(arg1, arg2): 
    if next(iter(arg2.keys())) != "dim":
        return 1.0
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    max_allowed_dim = arg1_value.ndim - 1
    min_allowed_dim = -1 * arg1_value.ndim
    return abs(arg2_value - max_allowed_dim)/MAX_N_DIM if arg2_value > max_allowed_dim else abs(min_allowed_dim - min(arg2_value, min_allowed_dim))/MAX_N_DIM

'''
    Corresponds to rule that asserts that arg1 and arg2 has the same number of dimensions. (Rule 3)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # 2d
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]])} # 2d
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # 2d
    arg2 = {"other_tensor": np.array([5])} # 1d, not the same as arg1
'''
def dist_3_rev(arg1, arg2):
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))    
    return abs(arg1_value.ndim-arg2_value.ndim)/MAX_N_DIM


'''
    Corresponds to rule that asserts that arg1 and arg2 have the same data type. (Rule 4)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.float32)} # float32
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int32)} # int32, not the same as arg1
'''
def dist_4_rev(arg1, arg2):
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    # dtypes of the two arguments
    dtype_1 = arg1_value.dtype
    dtype_2 = arg2_value.dtype
    ind_1 = list_of_available_dtypes.index(dtype_1)
    ind_2 = list_of_available_dtypes.index(dtype_2)
    total = len(list_of_available_dtypes)
    return abs(ind_1 - ind_2) / (total - 1)

'''
    Corresponds to rule that arg2 (index) is within the range of dimensions of arg1 (input tensor). (Rule 5)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"index": np.array([0, 1])} # Valid index
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"index": np.array([2])} # Invalid index, out of range
'''
def dist_5_rev(arg1, arg2):
    w1 = 0.6
    w2 = 0.4
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    if arg2_value.size == 0:
        min_val_arg2, max_val_arg2 = 0, 0 # empty index
    else:
        min_val_arg2, max_val_arg2 = np.min(arg2_value), np.max(arg2_value)
    shape_1 = list(arg1_value.shape)
    # calculating how smaller the min index is than 0
    # using MAX_SZ_NUM since the mutator/generator might not know proper limits for index like they know for tensor shapes since index is just another tensor
    component_1 = abs(0 - min(0, min_val_arg2))/MAX_SZ_NUM
    component_2 = 0
    for d in range(0, arg1_value.ndim):
        # calculating how bigger the max value is than the size of each dim
        component_2 += abs(shape_1[d]-max(shape_1[d], max_val_arg2+1))/MAX_SZ_NUM
    return w1*component_1 + w2*component_2

'''
    Corresponds to rule that asserts that the last dimension of arg1 is equal to the first dimension of arg2. (Rule 6)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]])} # Shape: (2, 2)
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"other_tensor": np.array([[6, 7]])} # Shape: (1, 2), first dim of arg2 (1) not equal to last dim of arg1 (2)
'''
def dist_6_rev(arg1, arg2):
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    # components of the objective function
    if arg1_value.ndim==0 or arg2_value.ndim==0:
        return 1.0
    l1 = arg1_value.shape
    l2 = arg2_value.shape
    return abs(l1[-1]-l2[0])/MAX_SZ_DIM

"""
Corresponds to rule that asserts that the size of the second dimenson of arg2 is equal to the size of the first dimension of arg1.

Positive Example:
arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
arg2 = {"other_tensor": np.array([[5, 6]])}         # Shape: (1, 2)


Negative Example:
arg1 = {"input_tensor": np.array([[1, 2], [3, 4], [5, 6]])} # Shape: (3, 2)
arg2 = {"other_tensor": np.array([[5, 6], [7, 8]])}         # Shape: (2, 2) second dim needs to be 3

Parameters:
arg1 (any): A dictionary containing the name of the first argument as key and the argument itself as value.
arg2 (any): A dictionary containing the name of the second argument as key and the argument itself as value.
"""
def dist_7_rev(arg1, arg2):
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    # components of the objective function
    if arg1_value.ndim < 1 or arg2_value.ndim < 2:
        return 1.0
    l1 = arg1_value.shape
    l2 = arg2_value.shape
    pos_l1 = 0 ## first
    pos_le = 2 ## second
    return abs(l1[0]-l2[1])/MAX_SZ_DIM

def dist_8_rev(arg1):
    """
        Corresponds to rule asserting that arg1 has an integer data type, i.e., np.int8, np.int16, np.int32, np.int64.

        Positive Example:
        arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int32)} # int32

        Negative Example:
        arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32, not int

        Parameters:
        arg1 (any): A dictionary containing the name of the first argument as key and the argument itself as value.
    """
    return dist_type(arg1, np.int8, np.uint8)

"""
Corresponds to rule asserting that arg is in 4D shape and all of its dimension sizes are positive.
"""
def dist_9_rev(arg):
    value = next(iter(arg.values()))
    if value.ndim != 4:
        return 1.0
    non_positive_dims = [dim for dim in value.shape if dim <= 0]
    if non_positive_dims:
        return 1.0
        # return sum(abs(dim) for dim in non_positive_dims) / (len(non_positive_dims) * MAX_SZ_DIM)
    return 0.0

"""
[conv_transpose2d] Corresponds to rule asserting that (stride * (input - 1) + weight - 2 * padding + output_padding) should be greater than zero for both height and width
"""
def dist_10_rev(arg1, arg2, arg3, arg4):
    expected_keys = ['input', 'weight', 'stride', 'padding']
    actual_keys = [list(arg.keys())[0] for arg in [arg1, arg2, arg3, arg4]]
    if actual_keys != expected_keys:
        return 1.0
    input_val = arg1['input']
    weight_val = arg2['weight']
    stride_val = arg3['stride']
    padding_val = arg4['padding']
    try:
        h_expr = stride_val * (input_val.shape[-2] - 1) + weight_val.shape[-2] - 2 * padding_val
        w_expr = stride_val * (input_val.shape[-1] - 1) + weight_val.shape[-1] - 2 * padding_val
        non_positive_exprs = [expr for expr in [h_expr, w_expr] if expr <= 0]
        if non_positive_exprs:
            return 1.0
            # return sum(abs(expr) for expr in non_positive_exprs) / (len(non_positive_exprs) * MAX_SZ_DIM)
        return 0.0
    except:
        return 1.0
    
def dist_11_rev(arg1, arg2, arg3):
    """
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
    """
    w1 = 0.3
    w2 = 0.7
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    arg3_value = next(iter(arg3.values()))
    if arg3_value.size == 0:
        min_val_arg3, max_val_arg3 = 0, 0 # empty index
    else:
        min_val_arg3, max_val_arg3 = np.min(arg3_value), np.max(arg3_value)
    shape_1 = list(arg1_value.shape)
    # calculating how smaller the min index is than 0
    # using MAX_SZ_NUM since the mutator/generator might not know proper limits for index like they know for tensor shapes since index is just another tensor
    component_1 = abs(0 - min(0, min_val_arg3))/MAX_SZ_NUM
    # check if dim is valid with dist 2
    d2_score = dist_2_rev(arg1, arg2)
    # calculating the distance between the dim size and the index value
    component_2 = abs(shape_1[arg2_value]-max(shape_1[arg2_value], max_val_arg3+1))/MAX_SZ_NUM if d2_score == 0 else 1
    return w1*component_1 + w2*component_2

def dist_12_rev(arg1, arg2):
    """
        Corresponds to a rule that ensures arg1 (low) is less than or equal to arg2 (high) only for primitive types (excludes tensors, list).
        
        Positive Example:
        arg1 = {"low": 2}
        arg2 = {"high": 5} # Valid range
        
        Negative Example:
        arg1 = {"low": 5}
        arg2 = {"high": 2} # Invalid range
    """
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    # Don't allow tensors or lists or tuples
    if isinstance(arg1_value, (np.ndarray, list, tuple)) or isinstance(arg2_value, (np.ndarray, list, tuple)):
        return 1.0
    
    return min(1, (arg1_value - arg2_value) / MAX_SZ_NUM) if arg1_value > arg2_value else 0.0

def dist_13_rev(arg1):
    """
        Corresponds to rule asserting that arg1 has an float data type.

        Positive Example:
        arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float16)} # float16

        Negative Example:
        arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.complex128)} # comlex128, not float

        Parameters:
        arg1 (any): A dictionary containing the name of the first argument as key and the argument itself as value.
    """
    return dist_type(arg1, np.float16, np.float64)

############### mapping ################

rule_to_distance = {
    1: {
        'rule_9': dist_9_rev,
        'rule_8': dist_8_rev,
        'rule_13': dist_13_rev
    },
    2: {
        'rule_1': dist_1_rev,
        'rule_2': dist_2_rev,
        'rule_3': dist_3_rev,
        'rule_4': dist_4_rev,
        'rule_5': dist_5_rev,
        'rule_6': dist_6_rev,
        'rule_7': dist_7_rev,
        'rule_12': dist_12_rev
    },
    3: {
        'rule_11': dist_11_rev
    },
    4: {
        'rule_10': dist_10_rev
    }
}

# Add rules where the order of arguments does not matter
# i.e. the nature of the arguments are the same
# e.g. two tensors having the same shape: does not matter if the first tensor is arg1 or arg2
# implication: do not check these rules for all permutations of the arguments
order_agnostic_rules = {
    1: ['rule_9', 'rule_8', 'rule_13'],  # arity 1 rules do not need to be added, but for completeness
    2: ['rule_1', 'rule_3', 'rule_4', 'rule_12']
}

'''
    Utility function. Returns distance to a range.

    Example:

    distance_to_range(0, 1, 4, 7) ~> 1/6
    distance_to_range(2, 1, 4, 7) ~> 0 (within range)
    distance_to_range(6, 1, 4, 7) ~> 2/6
'''
def distance_to_range(pos, lo, hi, total):
    if (pos >= lo and pos <= hi):
        dis = 0 # it is an integer
    ## not an integer
    elif pos < lo: # is it in a lower position?
        dis = abs(pos - lo) / (total - 1)    
    else: # must be in a position above hi.
        dis = abs(pos - hi) / (total - 1)    
    return dis
def dist_type(arg, low, high):
    """
        Corresponds to a rule that ensures the tensor argument is of a specific type (e.g., int, float).
        
        Positive Example:
        arg = {"value": np.array([1,2]).astype(np.int32)} # int
        lo = np.int8
        high = np.int64
        
        Negative Example:
        arg = {"value": np.array([1,2]).astype(np.float32)} # float
        lo = np.int8
        high = np.int64
    """
    # report distance to integer types  
    arg_value = next(iter(arg.values()))
    # only check for tensors
    if not isinstance(arg_value, np.ndarray):
        return 1.0
    
    # dtypes of the two arguments
    total = len(list_of_available_dtypes)
    dtype = arg_value.dtype
    ind = list_of_available_dtypes.index(dtype)
    lo = list_of_available_dtypes.index(low)
    hi = list_of_available_dtypes.index(high)
    return distance_to_range(ind, lo, hi, total)
'''
    Utility function, given an input, checks which rules it satisfies
'''
def check_rules(input_dict, print_rules=False):
    set_of_rules_passed = set()
    
    if len(input_dict.keys()) < 1:
        print("Not enough arguments to check rules")
        return set_of_rules_passed
    
    # Generate unique combinations of keys
    for arity in rule_to_distance:
        if len(input_dict.keys()) < arity:
            continue
        for args in combinations(input_dict.keys(), arity):
            for rule_name, distance_function in rule_to_distance[arity].items():
                # Check if the rule is satisfied for the current combination of arguments
                # i.e. distance function returns zero
                try:
                    arg_dicts = tuple({k: input_dict[k]} for k in args)
                    if distance_function(*arg_dicts) == 0:
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
                            if distance_function(*arg_dicts) == 0:
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
