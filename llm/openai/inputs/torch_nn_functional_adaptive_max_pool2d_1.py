
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    input_arr = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    input_dict = {"input": input_arr, "output_size": (2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 5, 7, dtype=torch.float32).sub(0.5).numpy()
    input_dict = {"input": input_arr, "output_size": (3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-1, 1, steps=27, dtype=torch.float32).reshape(3, 3, 3).numpy()
    input_dict = {"input": input_arr, "output_size": (1, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 2, 8, 8, dtype=torch.double).numpy()
    input_dict = {"input": input_arr, "output_size": (9, 9)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 10, 5, dtype=torch.double).numpy()
    input_dict = {"input": input_arr, "output_size": (10, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 3, 11, 13, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "output_size": (6, 7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(2 * 4 * 6, dtype=torch.float16).reshape(2, 4, 6).numpy()
    input_dict = {"input": input_arr, "output_size": (2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 4, 4, dtype=torch.float32).numpy()
    input_arr[0, 0, 0, 0] = np.nan
    input_arr[1, 0, 3, 3] = np.inf
    input_dict = {"input": input_arr, "output_size": (4, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 5, 2, 2, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (1, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(4, 7, 9, dtype=torch.float32) * -5.0).numpy()
    input_dict = {"input": input_arr, "output_size": (7, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 13, 17, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (3, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 3, 20, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (1, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool2d_1"] = adaptive_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_max_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool2d_1'.")


check_valid('torch.nn.functional.adaptive_max_pool2d', generated_inputs['torch.nn.functional.adaptive_max_pool2d_1'], lib="torch", suffix=1)
