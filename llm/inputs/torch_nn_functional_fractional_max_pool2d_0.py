
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fractional_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 1, 5, 5).astype(np.float32)
    kernel_size = (2, 2)
    output_size = (3, 3)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
    kernel_size = (3, 3)
    output_size = None
    output_ratio = (0.5, 0.5)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 3, 7, 7).astype(np.float32)
    kernel_size = (2, 2)
    output_size = (5, 5)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 1, 12, 12).astype(np.float32)
    kernel_size = (4, 4)
    output_size = None
    output_ratio = (0.6, 0.6)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(3, 1, 8, 8).astype(np.float32)
    kernel_size = (3, 3)
    output_size = (6, 6)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 5, 9, 9).astype(np.float32)
    kernel_size = (2, 2)
    output_size = None
    output_ratio = (0.7, 0.7)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(2, 2, 6, 6).astype(np.float32)
    kernel_size = (2, 1)
    output_size = (4, 3)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 1, 11, 11).astype(np.float32)
    kernel_size = (1, 2)
    output_size = None
    output_ratio = (0.4, 0.8)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(4, 1, 7, 7).astype(np.float32)
    kernel_size = (2, 2)
    output_size = (5, 5)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 4, 10, 10).astype(np.float32)
    kernel_size = (3, 3)
    output_size = None
    output_ratio = (0.55, 0.55)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for input_dict in list_of_inputs:
        if input_dict["output_size"] is None and input_dict["output_ratio"] is not None:
            input_dict["output_size"] = ()
        if input_dict["output_size"] is not None and input_dict["output_ratio"] is None:
            input_dict["output_ratio"] = ()


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.fractional_max_pool2d"] = fractional_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.fractional_max_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.fractional_max_pool2d'.")

check_valid('torch.nn.functional.fractional_max_pool2d', generated_inputs['torch.nn.functional.fractional_max_pool2d'], lib="torch", suffix=0)
