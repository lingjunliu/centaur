
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    input_arr = np.random.randn(1, 64, 8, 9).astype(np.float32)
    input_dict = {"output_size": (5, 7), "return_indices": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 3, 15, 7).astype(np.float64)
    input_dict = {"output_size": (5, 3), "return_indices": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(1, 64, 10, 9).astype(np.float32)
    input_dict = {"output_size": (7, 7), "return_indices": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(3, 32, 32).astype(np.float32)
    input_dict = {"output_size": (8, 8), "return_indices": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (np.random.rand(4, 1, 5, 5).astype(np.float32) - 0.5) * 100.0
    input_dict = {"output_size": (3, 3), "return_indices": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 5, 64, 64).astype(np.float32)
    input_dict = {"output_size": (1, 1), "return_indices": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(1, 2, 6, 13).astype(np.float32)
    input_dict = {"output_size": (6, 5), "return_indices": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(1, 1, 7, 5).astype(np.float64)
    input_dict = {"output_size": (7, 5), "return_indices": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(1 * 5 * 12 * 3, dtype=np.float32).reshape(1, 5, 12, 3) - 50.0
    input_dict = {"output_size": (4, 2), "return_indices": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(1, 4, 9, 1).astype(np.float32)
    input_dict = {"output_size": (3, 1), "return_indices": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.zeros((1, 1, 2, 2), dtype=np.float32)
    input_dict = {"output_size": (2, 2), "return_indices": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(8, 3, 17, 19).astype(np.float32)
    input_dict = {"output_size": (4, 4), "return_indices": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveMaxPool2d_2"] = adaptive_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveMaxPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool2d_2'.")


check_valid('torch.nn.AdaptiveMaxPool2d', generated_inputs['torch.nn.AdaptiveMaxPool2d_2'], lib="torch", suffix=2)
