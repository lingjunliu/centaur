
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def erfinv_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = np.array([0.1, 0.5, 0.9])
    out1 = np.array([])
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor
    input2 = np.array([[0.2, 0.6], [0.4, 0.8]])
    out2 = np.array([])
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tensor with negative values (but within the valid range)
    input3 = np.array([-0.2, -0.6, 0.3, 0.7])
    out3 = np.array([])
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with values close to 1 and -1
    input4 = np.array([-0.99, -0.9, 0.9, 0.99])
    out4 = np.array([])
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Empty tensor
    input5 = np.array([])
    out5 = np.array([])
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.erfinv"] = erfinv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erfinv'.")

check_valid('torch.special.erfinv', generated_inputs['torch.special.erfinv'], lib="torch", suffix=0)
