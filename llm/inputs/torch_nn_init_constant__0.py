
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def constant_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.zeros((3, 4), dtype=np.float32)
    val = 0.0
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.ones((2, 2), dtype=np.float64)
    val = 1.0
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.random.rand(5, 5).astype(np.float32)
    val = 2.5
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.empty((1, 10), dtype=np.float64)
    val = -1.0
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    val = 10.0
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    tensor = np.random.randn(2, 3, 4).astype(np.float32)
    val = 0.5
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    val = -2.0
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.zeros((1, 1, 1, 1), dtype=np.float32)
    val = 5.0
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    tensor = np.ones((4), dtype=np.float64)
    val = 100.0
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    val = 0.123
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    tensor = np.empty((0), dtype=np.float64)
    val = 2.7
    input_dict = {"tensor": tensor, "val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.constant_"] = constant_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.constant_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.constant_'.")

check_valid('torch.nn.init.constant_', generated_inputs['torch.nn.init.constant_'], lib="torch", suffix=0)
