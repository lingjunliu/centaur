import numpy as np
import logging
from itertools import combinations

from .definitions import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes

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
        component_2 += abs(shape_1[d]-max(shape_1[d], max_val_arg2))/MAX_SZ_NUM
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
    if arg1_value.ndim==0 or arg2_value.ndim==0:
        return 1.0
    l1 = arg1_value.shape
    l2 = arg2_value.shape
    pos_l1 = 0 ## first
    pos_le = 2 ## second
    return abs(l1[0]-l2[1])/MAX_SZ_DIM

"""
Corresponds to rule asserting that arg1 and arg2 have the **same** integer data types, i.e., np.int8, np.int16, np.int32, np.int64.

Positive Example:
arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int32)} # int32
arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int32)} # int32

Negative Example:
arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32, not int
arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int32)} # int32

Parameters:
arg1 (any): A dictionary containing the name of the first argument as key and the argument itself as value.
arg2 (any): A dictionary containing the name of the second argument as key and the argument itself as value.
"""
def dist_8_rev(arg1, arg2):
    # types are the same. pick one and report distance to integer types  
    arg1_value = next(iter(arg1.values()))
    arg2_value = next(iter(arg2.values()))
    # dtypes of the two arguments
    total = len(list_of_available_dtypes)
    dtype_1 = arg1_value.dtype
    dtype_2 = arg2_value.dtype        
    ind_1 = list_of_available_dtypes.index(dtype_1)
    ind_2 = list_of_available_dtypes.index(dtype_2)
    lo = list_of_available_dtypes.index(np.int8)
    hi = list_of_available_dtypes.index(np.int64)
    return (dist_4_rev(arg1, arg2) + distance_to_range(ind_1, lo, hi, total) + distance_to_range(ind_2, lo, hi, total))/3

############### mapping ################

rule_to_distance = {
    'rule_1': dist_1_rev,
    'rule_2': dist_2_rev,
    'rule_3': dist_3_rev,
    'rule_4': dist_4_rev,
    'rule_5': dist_5_rev,
    'rule_6': dist_6_rev,
    'rule_7': dist_7_rev,
    'rule_8': dist_8_rev,
}

order_agnostic_rules = ['rule_1', 'rule_3', 'rule_4', 'rule_8']

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
'''
    Utility function, given an input, checks which rules it satisfies
'''
def check_rules(input_dict, print_rules=False):
    # TODO: Add support for rules that take more than 2 arguments
    set_of_rules_passed = set()
    
    if len(input_dict.keys()) < 2:
        print("Not enough arguments to check rules")
        return set_of_rules_passed
    
    # Generate unique pairs of keys
    for arg_1, arg_2 in combinations(input_dict.keys(), 2):
        for rule_name, distance_function in rule_to_distance.items():
            # Check if the rule is satisfied for the current pair of arguments
            # i.e. distance function returns zero
            try:
                if distance_function({arg_1: input_dict[arg_1]}, {arg_2: input_dict[arg_2]}) == 0:
                    # If the rule is satisfied, add it to the set of passed rules
                    set_of_rules_passed.add((rule_name, arg_1, arg_2))
            except:
                pass    # The rule is not applicable

            # Change order of arguments to check the rule in the opposite direction unless the rule is order-agnostic
            if rule_name not in order_agnostic_rules:
                try:
                    if rule({arg_2: input_dict[arg_2]}, {arg_1: input_dict[arg_1]}):
                    # If the rule is satisfied, add it to the set of passed rules
                        set_of_rules_passed.add((rule_name, arg_2, arg_1))
                except:
                    pass    # The rule is not applicable
    # Optionally print the rules that have been passed
    if print_rules:
        for rule in set_of_rules_passed:
            print(f"Rule {rule[0]} passed between {rule[1]} and {rule[2]}")
    # Return the set of rules that have been passed
    return set_of_rules_passed