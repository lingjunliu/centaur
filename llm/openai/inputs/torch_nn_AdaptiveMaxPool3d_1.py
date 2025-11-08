
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.random.randn(1, 1, 4, 4, 4).astype(np.float32)
    input_dict = {
        "output_size": 2,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.random.randn(2, 3, 8, 10, 12).astype(np.float64)
    input_dict = {
        "output_size": 3,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (no batch dim, negative values)
    input_arr = (-np.random.rand(2, 5, 7, 9)).astype(np.float32)
    input_dict = {
        "output_size": 1,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (large channels as in docs)
    input_arr = torch.randn(1, 64, 8, 9, 10).numpy()
    input_dict = {
        "output_size": 7,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (output size > some input dims)
    input_arr = np.linspace(-1, 1, 4 * 1 * 3 * 5 * 7, dtype=np.float32).reshape(4, 1, 3, 5, 7)
    input_dict = {
        "output_size": 5,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (singleton spatial dims)
    input_arr = np.array([[[[[3.0]]]]], dtype=np.float32)  # (1,1,1,1,1)
    input_dict = {
        "output_size": 1,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (float16)
    input_arr = np.random.randn(3, 2, 6, 6, 6).astype(np.float16)
    input_dict = {
        "output_size": 4,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (transposed to get non-default strides)
    base = np.random.randn(2, 4, 3, 6, 5).astype(np.float32)
    input_arr = np.transpose(base, (0, 2, 1, 3, 4))  # (2,3,4,6,5)
    input_dict = {
        "output_size": 2,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (large H, W)
    input_arr = np.random.randn(1, 1, 2, 64, 32).astype(np.float32)
    input_dict = {
        "output_size": 2,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (no batch dim, thin H dimension)
    input_arr = np.random.randn(5, 10, 1, 3).astype(np.float32)
    input_dict = {
        "output_size": 3,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (W=1)
    input_arr = np.random.uniform(-100, 100, (2, 5, 13, 7, 1)).astype(np.float32)
    input_dict = {
        "output_size": 4,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (sequential values)
    input_arr = np.arange(2 * 1 * 9 * 9 * 9, dtype=np.float32).reshape(2, 1, 9, 9, 9)
    input_dict = {
        "output_size": 3,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveMaxPool3d_1"] = adaptive_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveMaxPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool3d_1'.")


check_valid('torch.nn.AdaptiveMaxPool3d', generated_inputs['torch.nn.AdaptiveMaxPool3d_1'], lib="torch", suffix=1)
