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
        s.add(And(*[Implies(i < v["arg1_ndim"], Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i)) for i in range(MAX_N_DIM)]))
    ),
    "rule_2": lambda s,v: (
        s.add(v["arg2_value"] >= -1 * v["arg1_ndim"]),
        s.add(v["arg2_value"] <= v["arg1_ndim"] - 1)
    ),
    "rule_3": lambda s,v: (
        s.add(v["arg1_ndim"] == v["arg2_ndim"])
    ),
    "rule_4": lambda s,v: (
        s.add(v["arg1_dtype"] == v["arg2_dtype"])
    ),
    "rule_5": lambda s,v: (
        s.add(Select(v["arg2_range"], 0) >= 0),
        s.add(And(*[Implies(i < v["arg1_ndim"], Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], i) - 1) for i in range(MAX_N_DIM)]))
    ),
    "rule_6": lambda s,v: (
        s.add(And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0)),
        s.add(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == Select(v["arg2_shape"], 0))
    ),
    "rule_7": lambda s,v: (
        s.add(And(v["arg1_ndim"] >= 1, v["arg2_ndim"] >= 2)),
        s.add(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 1))
    ),
    "rule_8": lambda s,v: (
        s.add(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5))  # check for integer types (1: np.int8, 5: np.uint8)
    ),
    "rule_9": lambda s,v: (
        s.add(v["arg1_ndim"] == 4),
        s.add(And(*[Implies(i < v["arg1_ndim"], Select(v["arg1_shape"], i) > 0) for i in range(MAX_N_DIM)]))
    ),
    "rule_10": lambda s,v: (
        s.add(And(v["arg1_ndim"] == 4, v["arg2_ndim"] == 4)),
        s.add(And(v["arg3_value"] >= 1, v["arg4_value"] >= 0)),
        s.add(v["arg3_value"] * (Select(v["arg1_shape"], v["arg1_ndim"] - 2) - 1) 
                        + Select(v["arg2_shape"], v["arg2_ndim"] - 2) - 2 * v["arg4_value"] > 0),
        s.add(v["arg3_value"] * (Select(v["arg1_shape"], v["arg1_ndim"] - 1) - 1) 
                        + Select(v["arg2_shape"], v["arg2_ndim"] - 1) - 2 * v["arg4_value"] > 0)
    ),
    "rule_11": lambda s,v: (
        s.add(Select(v["arg3_range"], 0) >= 0),
        s.add(Select(v["arg3_range"], 1) <= Select(v["arg1_shape"], v["arg2_value"]) - 1)
    ),
    "rule_12": lambda s,v: (
        s.add(v["arg1_value"] <= v["arg2_value"])
    ),
    "rule_13": lambda s,v: (
        s.add(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8))  # check for float types (6: np.float16, 8: np.float64)
    ),
    "rule_14": lambda s,v: (
        s.add(Or(*[And(i < v["arg1_ndim"], Select(v["arg1_shape"], i) > 0) for i in range(MAX_N_DIM)]))
    ),
    "rule_15": lambda s,v: (
        s.add(And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0)),
        s.add(And(*[Or(v["arg1_ndim"] - i < 0, v["arg2_ndim"] - i < 0, 
                       And(v["arg1_ndim"] - i >= 0, v["arg2_ndim"] - i >= 0, 
                           Or(Select(v["arg1_shape"], v["arg1_ndim"] - i) == 1, Select(v["arg2_shape"], v["arg2_ndim"] - i) == 1, 
                              Select(v["arg1_shape"], v["arg1_ndim"] - i) == Select(v["arg2_shape"], v["arg2_ndim"] - i))))
                    for i in range(MAX_N_DIM)]))
    ),
    "rule_16": lambda s,v: (
        s.add(Or(And(v["arg1_ndim"] >= 2, v["arg2_ndim"] >= 2, 
                     Select(v["arg1_shape"], v["arg1_ndim"] - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - 2)),
                 And(v["arg1_ndim"] >= 2, v["arg2_ndim"] == 1,
                     Select(v["arg1_shape"], v["arg1_ndim"] - 1) == Select(v["arg2_shape"], 0))))
    ),
    "rule_17": lambda s,v: (
        s.add(v["arg1_value"] >= 0)
    ),
    "rule_18": lambda s,v: (
        s.add(Select(v["arg1_range"], 0) >= 0)
    ),
    "rule_19": lambda s, v: (
        s.add(If(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5),
                 And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 5),
                 If(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8),
                    And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 8), True)))
    ),
    "rule_20": lambda s,v: (
        s.add(v["arg1_ndim"] == 1)
    ),
    "rule_21": lambda s,v: (
        s.add(v["arg1_value"] != 0)
    ),
    "rule_22": lambda s, v: (
        s.add(If(v["arg1_value"] > 0,
                 v["arg2_value"] <= v["arg3_value"],
                 If(v["arg1_value"] < 0,
                    v["arg2_value"] >= v["arg3_value"], True)))  
    ),
    "rule_23": lambda s,v: (
        s.add(v["arg1_ndim"] == 3),
        s.add(And(*[Implies(i < v["arg1_ndim"], Select(v["arg1_shape"], i) > 0) for i in range(MAX_N_DIM)]))
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
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))    

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        arg1_shape, arg2_shape = Array('arg1_shape', IntSort(), IntSort()), Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 1
        _(solver, 'rule_1', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 
                             'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 1
        _(solver, 'rule_1', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 
                             'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']})

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
    param2 = next(iter(arg2.keys()))
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    
    # Invariant learning phase
    if not solver: 
        if param2 != "dim" or not isinstance(arg1, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_value = Ints('arg1_ndim arg2_value')
    
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2)) 

        # Constraints for rule 2
        _(solver, 'rule_2', {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 2
        _(solver, 'rule_2', {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']})

'''
    Corresponds to rule that asserts that arg1 and arg2 has the same number of dimensions. (Rule 3)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # 2d
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]])} # 2d
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # 2d
    arg2 = {"other_tensor": np.array([5])} # 1d, not the same as arg1
'''

def rule_3_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))    

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 3
        _(solver, 'rule_3', {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 3
        _(solver, 'rule_3', {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']})

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
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    
    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_dtype, arg2_dtype = Ints('arg1_dtype arg2_dtype')
    
        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 4
        _(solver, 'rule_4', {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 4
        _(solver, 'rule_4', {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']})

'''
    Corresponds to rule that arg2 (index) is within the range of dimensions of arg1 (input tensor). (Rule 5)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"index": np.array([0, 1])} # Valid index
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"index": np.array([2])} # Invalid index, out of range
'''

def rule_5_func(arg1, arg2, solver=None):
    param2 = next(iter(arg2.keys()))
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver: 
        if param2 != "index" or not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_ndim = Int('arg1_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
    
        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_ndim == arg1.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 5
        _(solver, 'rule_5', {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 5
        _(solver, 'rule_5', {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 
                             'arg2_range': arg2['range']})

'''
    Corresponds to rule that asserts that the last dimension of arg1 is equal to the first dimension of arg2. (Rule 6)
    
    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]])} # Shape: (2, 2)
    
    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"other_tensor": np.array([[6, 7]])} # Shape: (1, 2), first dim of arg2 (1) not equal to last dim of arg1 (2)
'''

def rule_6_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        arg1_shape, arg2_shape = Array('arg1_shape', IntSort(), IntSort()), Array('arg2_shape', IntSort(), IntSort())
    
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        arg1_shape = Store(arg1_shape, arg1.ndim-1, arg1.shape[arg1.ndim-1])
        arg2_shape = Store(arg2_shape, 0, arg2.shape[0])
        
        # Constraints for rule 6
        _(solver, 'rule_6', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 
                             'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 6
        _(solver, 'rule_6', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 
                             'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']})

"""
    Corresponds to rule that asserts that the size of the second dimenson of arg2 is equal 
    to the size of the first dimension of arg1. (Rule 7)

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

def rule_7_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        arg1_shape, arg2_shape = Array('arg1_shape', IntSort(), IntSort()), Array('arg2_shape', IntSort(), IntSort())
    
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        arg1_shape = Store(arg1_shape, 0, arg1.shape[0])
        arg2_shape = Store(arg2_shape, 1, arg2.shape[1])
        
        # Constraints for rule 7
        _(solver, 'rule_7', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 
                             'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 7
        _(solver, 'rule_7', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 
                             'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']})

"""
    Corresponds to rule asserting that arg1 has an integer data type, i.e., np.int8, np.int16, np.int32, np.int64. (Rule 8)

    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int32)} # int32

    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32, not int

    Parameters:
    arg1 (any): A dictionary containing the name of the first argument as key and the argument itself as value.
"""

def rule_8_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False 
        
        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        
        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 8
        _(solver, 'rule_8', {'arg1_dtype': arg1_dtype}) 
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 8
        _(solver, 'rule_8', {'arg1_dtype': arg1['dtype']})

"""
    Corresponds to rule asserting that arg1 is in 4D shape and all of its dimension sizes are positive. (Rule 9)
"""

def rule_9_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort()) 

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 9
        _(solver, 'rule_9', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 9
        _(solver, 'rule_9', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}) 

"""
    [conv_transpose2d] Corresponds to rule asserting that (stride * (input - 1) + weight - 2 * padding + output_padding) 
                       should be greater than zero for both height and width. (Rule 10).
                       arg1 = input, arg2 = weight, arg3 = stride, arg4 = padding
"""

def rule_10_func(arg1, arg2, arg3, arg4, solver=None):
    param1 = next(iter(arg1.keys()))
    param2 = next(iter(arg2.keys()))
    param3 = next(iter(arg3.keys()))
    param4 = next(iter(arg4.keys()))
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver: 
        if (
            param1 != "input" or param2 != "weight" or 
            param3 != "stride" or param4 != "padding" or 
            not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray)
        ):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        arg1_shape, arg2_shape = Array('arg1_shape', IntSort(), IntSort()), Array('arg2_shape', IntSort(), IntSort())
        arg3_value, arg4_value = Ints('arg3_value, arg4_value')
        
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        arg1_shape = Store(arg1_shape, arg1.ndim-1, arg1.shape[arg1.ndim-1])
        arg1_shape = Store(arg1_shape, arg1.ndim-2, arg1.shape[arg1.ndim-2])
        arg2_shape = Store(arg2_shape, arg2.ndim-1, arg2.shape[arg2.ndim-1])
        arg2_shape = Store(arg2_shape, arg2.ndim-2, arg2.shape[arg2.ndim-2])
        solver.add(arg3_value == int(arg3)) 
        solver.add(arg4_value == int(arg4)) 
        
        # Constraints for rule 10
        _(solver, 'rule_10', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 
                              'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape,
                              'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 10
        _(solver, 'rule_10', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 
                              'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'],
                              'arg3_value': arg3['value'], 'arg4_value': arg4['value']})

'''
    Corresponds to a rule that ensures the index tensor (arg3) is within
    the constraint of the dimension size of the input tensor (arg1) along
    the specified dim (arg2). (Rule 11)
        
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
    param2 = next(iter(arg2.keys()))
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    
    # Invariant learning phase
    if not solver: 
        if param2 != "dim" or not isinstance(arg1, np.ndarray) or not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
    
        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 11
        _(solver, 'rule_11', {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 11
        _(solver, 'rule_11', {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 
                              'arg3_range': arg3['range']})

"""
    Corresponds to a rule that ensures arg1 (low) is less than or equal to arg2 (high) 
    only for primitive types (excludes tensors, list). (Rule 12)
        
    Positive Example:
    arg1 = {"low": 2}
    arg2 = {"high": 5} # Valid range
        
    Negative Example:
    arg1 = {"low": 5}
    arg2 = {"high": 2} # Invalid range
"""

def rule_12_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver: 
        if (
            not isinstance(arg1, (int, float, np.integer, np.floating)) or isinstance(arg1, bool) or 
            not isinstance(arg2, (int, float, np.integer, np.floating)) or isinstance(arg2, bool)
        ):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value, arg2_value = Reals('arg1_value arg2_value')
    
        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        
        # Constraints for rule 12
        _(solver, 'rule_12', {'arg1_value': arg1_value, 'arg2_value': arg2_value}) 
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 12
        _(solver, 'rule_12', {
            'arg1_value': ToReal(arg1['value']) if is_int_value(arg1['value']) else arg1['value'],
            'arg2_value': ToReal(arg2['value']) if is_int_value(arg2['value']) else arg2['value']})

"""
    Corresponds to rule asserting that arg1 has an float data type. (Rule 13)

    Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float16)} # float16

    Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.complex128)} # comlex128, not float

    Parameters:
    arg1 (any): A dictionary containing the name of the first argument as key and the argument itself as value.
"""

def rule_13_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        
        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 13
        _(solver, 'rule_13', {'arg1_dtype': arg1_dtype}) 
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 13
        _(solver, 'rule_13', {'arg1_dtype': arg1['dtype']})

"""
    Corresponds to rule asserting that arg1 (input_tensor) should not be empty. (Rule 14)
"""

def rule_14_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
    
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 14
        _(solver, 'rule_14', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 14
        _(solver, 'rule_14', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']})

"""
    Corresponds to rule for the broadcasting semantics. (Rule 15)
"""

def rule_15_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))    

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        arg1_shape, arg2_shape = Array('arg1_shape', IntSort(), IntSort()), Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 15
        _(solver, 'rule_15', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 
                              'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 15
        _(solver, 'rule_15', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 
                              'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']})

"""
    Corresponds to rule for shape alignment for matrix multiplication. (Rule 16)
"""

def rule_16_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray) or not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim, arg2_ndim = Ints('arg1_ndim arg2_ndim')
        arg1_shape, arg2_shape = Array('arg1_shape', IntSort(), IntSort()), Array('arg2_shape', IntSort(), IntSort())
    
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        arg1_shape = Store(arg1_shape, arg1.ndim - 1, arg1.shape[-1])
        arg2_shape = Store(arg2_shape, arg2.ndim - 1, arg2.shape[-1])
        if arg2.ndim >= 2:
            arg2_shape = Store(arg2_shape, arg2.ndim - 2, arg2.shape[-2])
        
        # Constraints for rule 16
        _(solver, 'rule_16', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 
                              'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 16
        _(solver, 'rule_16', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 
                              'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']})

"""
    Corresponds to rule asserting that variable (primitive) has to be non-negative. (Rule 17)
"""

def rule_17_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, (int, float, np.integer, np.floating)) or isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
    
        # Value assignments
        solver.add(arg1_value == arg1)
        
        # Constraints for rule 17
        _(solver, 'rule_17', {'arg1_value': arg1_value}) 
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 17
        _(solver, 'rule_17', {'arg1_value': arg1['value']}) 

"""
    Corresponds to rule asserting that arg1 (input_tensor) only contains non negative values. (Rule 18)
"""

def rule_18_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
    
        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))

        # Constraints for rule 18
        _(solver, 'rule_18', {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 18
        _(solver, 'rule_18', {'arg1_range': arg1['range']})

"""
    Corresponds to rule asserting that
    if a tensor (arg1) is integer, another non tensor (arg2) must be integer, and
    if a tensor (arg1) is float, another non tensor (arg2) must be either integer or float. (Rule 19)
"""

def rule_19_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    
    # Invariant learning phase
    if not solver: 
        if (
            not isinstance(arg1, np.ndarray) or 
            not isinstance(arg2, (int, float, np.integer, np.floating)) or isinstance(arg2, bool)
        ):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_dtype, arg2_dtype = Ints('arg1_dtype arg2_dtype')
    
        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        if isinstance(arg2, int):
            solver.add(arg2_dtype == 4)  # index of np.int64
        else: 
            solver.add(arg2_dtype == 8)  # index of np.float64

        # Constraints for rule 19
        _(solver, 'rule_19', {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 19
        _(solver, 'rule_19', {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']})

"""
    Corresponds to rule asserting that arg1 (input_tensor) needs to be 1 dimenstional. (Rule 20)
"""

def rule_20_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False 

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        
        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 20
        _(solver, 'rule_20', {'arg1_ndim': arg1_ndim}) 
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 20
        _(solver, 'rule_20', {'arg1_ndim': arg1['ndim']})

"""
    Corresponds to rule asserting that variable (primitive) should not be zero. (Rule 21)
"""

def rule_21_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, (int, float, np.integer, np.floating)) or isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
    
        # Value assignments
        solver.add(arg1_value == arg1)
        
        # Constraints for rule 21
        _(solver, 'rule_21', {'arg1_value': arg1_value}) 
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 21
        _(solver, 'rule_21', {'arg1_value': arg1['value']}) 

"""
    [arange] Corresponds to rule asserting that for primitive type variables (arg1, arg2, arg3),
             if arg1 is greater than zero, arg2 should be smaller than or equal to arg3, and
             if arg1 is smaller than zero, arg2 should be greater than or equal to arg3. (Rule 22)
"""

def rule_22_func(arg1, arg2, arg3, solver=None):
    param1 = next(iter(arg1.keys()))
    param2 = next(iter(arg2.keys()))
    param3 = next(iter(arg3.keys()))
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if (
            param1 != "step" or param2 != "start" or param3 != "end" or 
            not isinstance(arg1, (int, float, np.integer, np.floating)) or isinstance(arg1, bool) or
            not isinstance(arg2, (int, float, np.integer, np.floating)) or isinstance(arg2, bool) or
            not isinstance(arg3, (int, float, np.integer, np.floating)) or isinstance(arg3, bool)
        ):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value, arg2_value, arg3_value = Reals('arg1_value arg2_value arg3_value')
    
        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        
        # Constraints for rule 22
        _(solver, 'rule_22', {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 22
        _(solver, 'rule_22', {
            'arg1_value': ToReal(arg1['value']) if is_int_value(arg1['value']) else arg1['value'],
            'arg2_value': ToReal(arg2['value']) if is_int_value(arg2['value']) else arg2['value'],
            'arg3_value': ToReal(arg3['value']) if is_int_value(arg3['value']) else arg3['value']})

"""
    Corresponds to rule asserting that arg1 is in 3D shape and all of its dimension sizes are positive. (Rule 23)
"""

def rule_23_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver: 
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort()) 

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 23
        _(solver, 'rule_23', {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        # Constraints for rule 23
        _(solver, 'rule_23', {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}) 

############### mapping ################

rule_func_map = { 
    1: {
        'rule_8': rule_8_func,
        'rule_9': rule_9_func,
        'rule_13': rule_13_func,
        'rule_14': rule_14_func,
        'rule_17': rule_17_func,
        'rule_18': rule_18_func,
        'rule_20': rule_20_func,
        'rule_21': rule_21_func,
        'rule_23': rule_23_func
    },
    2: {
        'rule_1': rule_1_func,
        'rule_2': rule_2_func,
        'rule_3': rule_3_func,
        'rule_4': rule_4_func,
        'rule_5': rule_5_func,
        'rule_6': rule_6_func,
        'rule_7': rule_7_func,
        'rule_12': rule_12_func,
        'rule_15': rule_15_func,
        'rule_16': rule_16_func,
        'rule_19': rule_19_func
    },
    3: {
        'rule_11': rule_11_func,
        'rule_22': rule_22_func
    },
    4: {
        'rule_10': rule_10_func
    }
}

# Add rules where the order of arguments does not matter
# i.e. the nature of the arguments are the same
# e.g. two tensors having the same shape: does not matter if the first tensor is arg1 or arg2
# implication: do not check these rules for all permutations of the arguments
order_agnostic_rules = {
    1: ['rule_8', 'rule_9', 'rule_13', 'rule_14', 'rule_17', 
        'rule_18', 'rule_20', 'rule_21', 'rule_23'],  # not necessary actually
    2: ['rule_1', 'rule_3', 'rule_4', 'rule_12', 'rule_15']
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
