
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def CompilationUnit_inputs():
    list_of_inputs = []

    # Input 1: Basic valid string
    input1 = """
    def foo(x):
        return x + 1
    """
    input_dict1 = {"script": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2:  More complex function with multiple operations
    input2 = """
    def bar(x, y):
        z = x * y
        return z - 2
    """
    input_dict2 = {"script": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Function with conditional statement
    input3 = """
    def baz(x):
        if x > 0:
            return x * 2
        else:
            return x / 2
    """
    input_dict3 = {"script": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Multiple functions defined in the same string
    input4 = """
    def func1(a):
        return a + 5

    def func2(b):
        return b * 3
    """
    input_dict4 = {"script": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5:  Function calling another function (defined within the same string)
    input5 = """
    def inner(x):
        return x * x

    def outer(y):
        return inner(y) + 1
    """
    input_dict5 = {"script": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = CompilationUnit_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('CompilationUnit', generated_inputs)
