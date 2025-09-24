
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def generate_glu_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D input, default dim
    input1 = np.random.randn(4, 2).astype(np.float32)
    input_dict1 = {"input": input1, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D input, dim=1
    input2 = np.random.randn(2, 4).astype(np.float32)
    input_dict2 = {"input": input2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D input, dim=1
    input3 = np.random.randn(1, 4, 2).astype(np.float32)
    input_dict3 = {"input": input3, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 3D input, dim=2
    input4 = np.random.randn(1, 2, 4).astype(np.float32)
    input_dict4 = {"input": input4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D input, dim=1
    input5 = np.random.randn(1, 4, 2, 2).astype(np.float32)
    input_dict5 = {"input": input5, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 4D input, dim=2
    input6 = np.random.randn(1, 2, 4, 2).astype(np.float32)
    input_dict6 = {"input": input6, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 4D input, dim=3
    input7 = np.random.randn(1, 2, 2, 4).astype(np.float32)
    input_dict7 = {"input": input7, "dim": 3}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 2D input, dim=0
    input8 = np.random.randn(4, 2).astype(np.float32)
    input_dict8 = {"input": input8, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 3D input, dim=0
    input9 = np.random.randn(4, 2, 2).astype(np.float32)
    input_dict9 = {"input": input9, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 2D input with negative values, default dim
    input10 = np.random.randn(4, 2).astype(np.float32) * -1
    input_dict10 = {"input": input10, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.GLU"] = generate_glu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.GLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GLU'.")

check_valid('torch.nn.GLU', generated_inputs['torch.nn.GLU'], lib="torch", suffix=0)
