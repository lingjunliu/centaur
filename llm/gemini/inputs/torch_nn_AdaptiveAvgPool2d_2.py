
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_nn_AdaptiveAvgPool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with H and W specified
    input1 = torch.randn(1, 3, 32, 32).numpy()
    output_size1 = (16, 16)
    input_dict1 = {"output_size": output_size1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Square output size
    input2 = torch.randn(1, 3, 64, 64).numpy()
    output_size2 = (8, 8)
    input_dict2 = {"output_size": output_size2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Batch size > 1
    input3 = torch.randn(4, 3, 28, 28).numpy()
    output_size3 = (14, 14)
    input_dict3 = {"output_size": output_size3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different input dimensions
    input4 = torch.randn(1, 64, 128, 256).numpy()
    output_size4 = (64, 128)
    input_dict4 = {"output_size": output_size4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Single channel input
    input5 = torch.randn(1, 1, 32, 32).numpy()
    output_size5 = (16, 16)
    input_dict5 = {"output_size": output_size5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Smaller input size than output size (will be upsampled)
    input6 = torch.randn(1, 3, 8, 8).numpy()
    output_size6 = (16, 16)
    input_dict6 = {"output_size": output_size6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Non-square input
    input7 = torch.randn(1, 3, 32, 64).numpy()
    output_size7 = (16, 32)
    input_dict7 = {"output_size": output_size7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Another different input size
    input8 = torch.randn(1, 3, 100, 50).numpy()
    output_size8 = (50, 25)
    input_dict8 = {"output_size": output_size8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Output size with same height as input
    input9 = torch.randn(1, 3, 32, 64).numpy()
    output_size9 = (32, 32)
    input_dict9 = {"output_size": output_size9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Larger batch size
    input10 = torch.randn(8, 3, 64, 64).numpy()
    output_size10 = (32, 32)
    input_dict10 = {"output_size": output_size10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Small output size
    input11 = torch.randn(1, 3, 128, 128).numpy()
    output_size11 = (1, 1)
    input_dict11 = {"output_size": output_size11, "input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveAvgPool2d_2"] = torch_nn_AdaptiveAvgPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveAvgPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool2d_2'.")

check_valid('torch.nn.AdaptiveAvgPool2d', generated_inputs['torch.nn.AdaptiveAvgPool2d_2'], lib="torch", suffix=2)
