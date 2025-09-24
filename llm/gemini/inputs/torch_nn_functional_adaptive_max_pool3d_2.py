
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_max_pool3d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input1 = np.random.rand(1, 1, 10, 10, 10).astype(np.float32)
    output_size1 = 5
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different input size
    input2 = np.random.rand(2, 3, 20, 20, 20).astype(np.float32)
    output_size2 = 7
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Smaller input, smaller output
    input3 = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    output_size3 = 2
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Non-square input
    input4 = np.random.rand(1, 1, 10, 15, 20).astype(np.float32)
    output_size4 = 3
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger input, larger output
    input5 = np.random.rand(4, 5, 30, 30, 30).astype(np.float32)
    output_size5 = 10
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: output_size = 1
    input6 = np.random.rand(1, 1, 10, 10, 10).astype(np.float32)
    output_size6 = 1
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Different dtype
    input7 = np.random.rand(1, 1, 10, 10, 10).astype(np.float64)
    output_size7 = 5
    input_dict7 = {"input": input7, "output_size": output_size7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Small input size
    input8 = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    output_size8 = 2
    input_dict8 = {"input": input8, "output_size": output_size8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Larger number of channels
    input9 = np.random.rand(5, 10, 15, 15, 15).astype(np.float32)
    output_size9 = 4
    input_dict9 = {"input": input9, "output_size": output_size9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Different output size
    input10 = np.random.rand(2, 2, 12, 12, 12).astype(np.float32)
    output_size10 = 6
    input_dict10 = {"input": input10, "output_size": output_size10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.adaptive_max_pool3d_2"] = adaptive_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.adaptive_max_pool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool3d_2'.")

check_valid('torch.nn.functional.adaptive_max_pool3d', generated_inputs['torch.nn.functional.adaptive_max_pool3d_2'], lib="torch", suffix=2)
