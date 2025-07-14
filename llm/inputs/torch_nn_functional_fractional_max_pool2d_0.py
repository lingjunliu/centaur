
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fractional_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 1, 32, 32).numpy()
    kernel_size = (5, 5)
    output_size = (16, 16)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": (0.5, 0.5) if output_ratio is None else output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 3, 64, 64).numpy()
    kernel_size = (3, 3)
    output_size = None
    output_ratio = (0.5, 0.5)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(2, 3, 128, 128).numpy()
    kernel_size = (2, 2)
    output_size = (64, 64)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": (0.5, 0.5) if output_ratio is None else output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 1, 256, 256).numpy()
    kernel_size = (4, 4)
    output_size = None
    output_ratio = (0.75, 0.75)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(4, 3, 32, 32).numpy()
    kernel_size = (2, 1)
    output_size = (10, 16)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": (0.5, 0.5) if output_ratio is None else output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(1, 1, 16, 16).numpy()
    kernel_size = (3, 2)
    output_size = None
    output_ratio = (0.6, 0.4)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(2, 3, 64, 32).numpy()
    kernel_size = (4, 3)
    output_size = (32, 16)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": (0.5, 0.5) if output_ratio is None else output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(1, 1, 128, 64).numpy()
    kernel_size = (2, 2)
    output_size = None
    output_ratio = (0.8, 0.25)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Example with negative values
    input_tensor = torch.randn(1, 1, 32, 32) - 1.0
    input_tensor = input_tensor.numpy()
    kernel_size = (5, 5)
    output_size = (16, 16)
    output_ratio = None
    return_indices = False
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": (0.5, 0.5) if output_ratio is None else output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(1, 3, 64, 64).numpy()
    kernel_size = (3, 3)
    output_size = None
    output_ratio = (0.25, 0.25)
    return_indices = True
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "output_size": output_size, "output_ratio": output_ratio, "return_indices": return_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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
