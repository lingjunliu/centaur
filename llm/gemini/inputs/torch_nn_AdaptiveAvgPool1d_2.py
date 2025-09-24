
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_1 = torch.randn(1, 64, 8).numpy()
    output_size_1 = (5,)
    input_dict_1 = {"output_size": output_size_1, "input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Different input size
    input_2 = torch.randn(2, 32, 16).numpy()
    output_size_2 = (7,)
    input_dict_2 = {"output_size": output_size_2, "input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Batch size of 1
    input_3 = torch.randn(1, 128, 32).numpy()
    output_size_3 = (10,)
    input_dict_3 = {"output_size": output_size_3, "input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Single channel
    input_4 = torch.randn(4, 1, 64).numpy()
    output_size_4 = (20,)
    input_dict_4 = {"output_size": output_size_4, "input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Input with length 1
    input_5 = torch.randn(2, 16, 1).numpy()
    output_size_5 = (1,)
    input_dict_5 = {"output_size": output_size_5, "input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Input with large values
    input_6 = (torch.rand(1, 3, 128) * 100).numpy()
    output_size_6 = (32,)
    input_dict_6 = {"output_size": output_size_6, "input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Input with small values
    input_7 = (torch.rand(1, 3, 128) * 0.01).numpy()
    output_size_7 = (32,)
    input_dict_7 = {"output_size": output_size_7, "input": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: C, Lin format
    input_8 = torch.randn(64, 8).numpy()
    output_size_8 = (5,)
    input_dict_8 = {"output_size": output_size_8, "input": input_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Different output size
    input_9 = torch.randn(2, 32, 16).numpy()
    output_size_9 = (1,)
    input_dict_9 = {"output_size": output_size_9, "input": input_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Input with different number of channels
    input_10 = torch.randn(4, 256, 64).numpy()
    output_size_10 = (20,)
    input_dict_10 = {"output_size": output_size_10, "input": input_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Input of different dtype
    input_11 = torch.randn(1, 64, 8, dtype=torch.float64).numpy()
    output_size_11 = (5,)
    input_dict_11 = {"output_size": output_size_11, "input": input_11}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveAvgPool1d_2"] = adaptive_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveAvgPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool1d_2'.")

check_valid('torch.nn.AdaptiveAvgPool1d', generated_inputs['torch.nn.AdaptiveAvgPool1d_2'], lib="torch", suffix=2)
