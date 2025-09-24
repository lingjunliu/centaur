
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def alphadropout_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input1 = np.random.randn(20, 16).astype(np.float32)
    input_dict1 = {"p": 0.2, "inplace": False, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: inplace=True
    input2 = np.random.randn(10, 5).astype(np.float32)
    input_dict2 = {"p": 0.3, "inplace": True, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different p value
    input3 = np.random.randn(5, 5, 5).astype(np.float32)
    input_dict3 = {"p": 0.7, "inplace": False, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D input
    input4 = np.random.randn(100).astype(np.float32)
    input_dict4 = {"p": 0.1, "inplace": False, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D input
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict5 = {"p": 0.4, "inplace": True, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: p=0.0
    input6 = np.random.randn(10, 10).astype(np.float32)
    input_dict6 = {"p": 0.0, "inplace": False, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: p=1.0
    input7 = np.random.randn(5, 8).astype(np.float32)
    input_dict7 = {"p": 1.0, "inplace": True, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Small input size
    input8 = np.random.randn(1, 1).astype(np.float32)
    input_dict8 = {"p": 0.5, "inplace": False, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Large input size
    input9 = np.random.randn(100, 100).astype(np.float32)
    input_dict9 = {"p": 0.6, "inplace": True, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Input with zeros
    input10 = np.zeros((10, 10)).astype(np.float32)
    input_dict10 = {"p": 0.2, "inplace": False, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Input with negative values
    input11 = np.random.randn(5, 5) - 2.0
    input11 = input11.astype(np.float32)
    input_dict11 = {"p": 0.3, "inplace": True, "input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AlphaDropout"] = alphadropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AlphaDropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AlphaDropout'.")

check_valid('torch.nn.AlphaDropout', generated_inputs['torch.nn.AlphaDropout'], lib="torch", suffix=0)
