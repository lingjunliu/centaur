
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def GELU_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"approximate": "none", "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict2 = {"approximate": "none", "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([0.0], dtype=np.float32)
    input_dict3 = {"approximate": "none", "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict4 = {"approximate": "tanh", "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict5 = {"approximate": "tanh", "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.randn(2, 3).astype(np.float32)
    input_dict6 = {"approximate": "none", "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.random.randn(2, 3).astype(np.float32)
    input_dict7 = {"approximate": "tanh", "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.randn(1, 5, 5).astype(np.float32)
    input_dict8 = {"approximate": "none", "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.randn(1, 5, 5).astype(np.float32)
    input_dict9 = {"approximate": "tanh", "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict10 = {"approximate": "none", "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11
    input11 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict11 = {"approximate": "tanh", "input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.GELU"] = GELU_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.GELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GELU'.")

check_valid('torch.nn.GELU', generated_inputs['torch.nn.GELU'], lib="torch", suffix=0)
