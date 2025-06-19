
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def fork_inputs():
    list_of_inputs = []

    def foo(a, b):
        return a + b

    input1 = {
        "func": foo,
        "*args": [np.array(1), 2],
        "**kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input1))

    def bar(a, b):
        return a * b

    input2 = {
        "func": bar,
        "*args": [np.array(4), 5],
        "**kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input2))

    def baz(a):
        return a * 2

    input3 = {
        "func": baz,
        "*args": [np.array([-1.0, -2.0])],
        "**kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input3))

    def simple_func(x):
        return x*x

    input4 = {
        "func": simple_func,
        "*args": [np.array(5)],
        "**kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input4))

    def add_arrays(x, y):
        return x + y

    input5 = {
        "func": add_arrays,
        "*args": [np.array([1,2,3]), np.array([4,5,6])],
        "**kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input5))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.fork"] = fork_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.fork' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.fork'.")

check_valid('torch.jit.fork', generated_inputs['torch.jit.fork'], lib="torch")
