
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_nn_AdaptiveAvgPool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    output_size = 7
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different batch size
    input = np.random.randn(4, 3, 32, 32).astype(np.float32)
    output_size = 7
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different number of channels
    input = np.random.randn(1, 64, 32, 32).astype(np.float32)
    output_size = 7
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different input size
    input = np.random.randn(1, 3, 64, 64).astype(np.float32)
    output_size = 7
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple output size
    input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    output_size = (5, 7)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different tuple output size
    input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    output_size = (10, 20)
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Smaller input size than output size
    input = np.random.randn(1, 3, 16, 16).astype(np.float32)
    output_size = 32
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single channel input
    input = np.random.randn(1, 1, 32, 32).astype(np.float32)
    output_size = 7
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large input size
    input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    output_size = 64
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: rectangular input
    input = np.random.randn(1, 3, 64, 32).astype(np.float32)
    output_size = 16
    input_dict = {"output_size": output_size, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveAvgPool2d_1"] = torch_nn_AdaptiveAvgPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveAvgPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool2d_1'.")

check_valid('torch.nn.AdaptiveAvgPool2d', generated_inputs['torch.nn.AdaptiveAvgPool2d_1'], lib="torch", suffix=1)
