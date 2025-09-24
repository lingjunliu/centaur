
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input1 = torch.randn(1, 3, 32, 32).numpy()
    output_size1 = 16
    return_indices1 = False
    input_dict1 = {"output_size": output_size1, "return_indices": return_indices1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Tuple output_size
    input2 = torch.randn(1, 3, 32, 32).numpy()
    output_size2 = (8, 16)
    return_indices2 = True
    input_dict2 = {"output_size": output_size2[0], "return_indices": return_indices2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Batch size > 1
    input3 = torch.randn(4, 3, 32, 32).numpy()
    output_size3 = 8
    return_indices3 = False
    input_dict3 = {"output_size": output_size3, "return_indices": return_indices3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Single channel input
    input4 = torch.randn(1, 1, 32, 32).numpy()
    output_size4 = (16, 16)
    return_indices4 = True
    input_dict4 = {"output_size": output_size4[0], "return_indices": return_indices4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different input dimensions
    input5 = torch.randn(1, 3, 64, 128).numpy()
    output_size5 = 32
    return_indices5 = False
    input_dict5 = {"output_size": output_size5, "return_indices": return_indices5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Small input size
    input6 = torch.randn(1, 3, 16, 16).numpy()
    output_size6 = 8
    return_indices6 = True
    input_dict6 = {"output_size": output_size6, "return_indices": return_indices6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Non-square input
    input7 = torch.randn(1, 3, 20, 40).numpy()
    output_size7 = 10
    return_indices7 = False
    input_dict7 = {"output_size": output_size7, "return_indices": return_indices7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: output size is bigger than input size
    input8 = torch.randn(1, 3, 8, 8).numpy()
    output_size8 = 16
    return_indices8 = True
    input_dict8 = {"output_size": output_size8, "return_indices": return_indices8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Larger batch size, different output size
    input9 = torch.randn(8, 3, 24, 24).numpy()
    output_size9 = 6
    return_indices9 = False
    input_dict9 = {"output_size": output_size9, "return_indices": return_indices9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Different channel size
    input10 = torch.randn(1, 64, 16, 16).numpy()
    output_size10 = 4
    return_indices10 = True
    input_dict10 = {"output_size": output_size10, "return_indices": return_indices10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Minimal input size
    input11 = torch.randn(1, 3, 2, 2).numpy()
    output_size11 = 1
    return_indices11 = False
    input_dict11 = {"output_size": output_size11, "return_indices": return_indices11, "input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveMaxPool2d_1"] = adaptive_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveMaxPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool2d_1'.")

check_valid('torch.nn.AdaptiveMaxPool2d', generated_inputs['torch.nn.AdaptiveMaxPool2d_1'], lib="torch", suffix=1)
