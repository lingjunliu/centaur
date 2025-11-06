
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    inp = torch.randn(1, 3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {"output_size": 4, "return_indices": False, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 1, 10, 7, dtype=torch.float64).numpy()
    input_dict = {"output_size": 5, "return_indices": True, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(64, 9, 9, dtype=torch.float32).numpy()
    input_dict = {"output_size": 3, "return_indices": False, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 3, 32, 24, dtype=torch.float32).numpy()
    input_dict = {"output_size": 6, "return_indices": True, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (torch.randn(1, 1, 5, 5, dtype=torch.float32) * 10 - 5).numpy()
    input_dict = {"output_size": 5, "return_indices": False, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 7, 9, dtype=torch.float32).numpy()
    input_dict = {"output_size": 7, "return_indices": False, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 8, 15, 15, dtype=torch.float64).numpy()
    input_dict = {"output_size": 1, "return_indices": True, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 16, 14, 10, dtype=torch.float32).numpy()
    input_dict = {"output_size": 2, "return_indices": False, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.zeros(1, 6, 6, dtype=torch.float32).numpy()
    input_dict = {"output_size": 2, "return_indices": True, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (torch.randn(5, 2, 11, 13, dtype=torch.float32) * 2 - 1).numpy()
    input_dict = {"output_size": 3, "return_indices": False, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 4, 9, 5, dtype=torch.float32).numpy()
    input_dict = {"output_size": 5, "return_indices": True, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 12, 7, dtype=torch.float64).numpy()
    input_dict = {"output_size": 7, "return_indices": False, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveMaxPool2d_1"] = adaptive_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveMaxPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool2d_1'.")


check_valid('torch.nn.AdaptiveMaxPool2d', generated_inputs['torch.nn.AdaptiveMaxPool2d_1'], lib="torch", suffix=1)
