
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1: Basic 5D input
    input1 = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    output_size1 = 5
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 4D input
    input2 = np.random.rand(3, 10, 10, 10).astype(np.float32)
    output_size2 = 7
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 5D input, different shape
    input3 = np.random.rand(2, 5, 15, 20, 25).astype(np.float32)
    output_size3 = 10
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Small input
    input4 = np.random.rand(1, 1, 4, 4, 4).astype(np.float32)
    output_size4 = 2
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different data type
    input5 = np.random.rand(1, 3, 10, 10, 10).astype(np.float64)
    output_size5 = 5
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Output size larger than input
    input6 = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    output_size6 = 7
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))


    # Input 8: 5D input, output size 1
    input8 = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    output_size8 = 1
    input_dict8 = {"input": input8, "output_size": output_size8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Large batch size
    input9 = np.random.rand(16, 3, 10, 10, 10).astype(np.float32)
    output_size9 = 5
    input_dict9 = {"input": input9, "output_size": output_size9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Output size same as input size
    input10 = np.random.rand(1, 1, 8, 8, 8).astype(np.float32)
    output_size10 = 8
    input_dict10 = {"input": input10, "output_size": output_size10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.adaptive_avg_pool3d_2"] = adaptive_avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool3d_2'.")

check_valid('torch.nn.functional.adaptive_avg_pool3d', generated_inputs['torch.nn.functional.adaptive_avg_pool3d_2'], lib="torch", suffix=2)
