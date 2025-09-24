
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(1, 64, 8).numpy()
    output_size1 = 5
    input_dict1 = {"output_size": output_size1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = torch.randn(2, 32, 16).numpy()
    output_size2 = 7
    input_dict2 = {"output_size": output_size2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = torch.randn(1, 1, 10).numpy()
    output_size3 = 3
    input_dict3 = {"output_size": output_size3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4 (C, Lin) format
    input4 = torch.randn(32, 20).numpy()
    output_size4 = 10
    input_dict4 = {"output_size": output_size4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5 (N, C, Lin)
    input5 = torch.randn(4, 16, 32).numpy()
    output_size5 = 16
    input_dict5 = {"output_size": output_size5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6 
    input6 = torch.randn(1, 3, 1).numpy()
    output_size6 = 1
    input_dict6 = {"output_size": output_size6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = torch.randn(8, 4, 64).numpy()
    output_size7 = 32
    input_dict7 = {"output_size": output_size7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8
    input8 = torch.randn(1, 256, 128).numpy()
    output_size8 = 64
    input_dict8 = {"output_size": output_size8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9 (C, Lin)
    input9 = torch.randn(64, 50).numpy()
    output_size9 = 25
    input_dict9 = {"output_size": output_size9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = torch.randn(2, 8, 4).numpy()
    output_size10 = 2
    input_dict10 = {"output_size": output_size10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11
    input11 = torch.randn(1, 128, 256).numpy()
    output_size11 = 128
    input_dict11 = {"output_size": output_size11, "input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveAvgPool1d_1"] = adaptive_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveAvgPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool1d_1'.")

check_valid('torch.nn.AdaptiveAvgPool1d', generated_inputs['torch.nn.AdaptiveAvgPool1d_1'], lib="torch", suffix=1)
