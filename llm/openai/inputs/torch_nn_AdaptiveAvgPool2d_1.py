
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptiveavgpool2d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(1, 64, 8, 9).numpy()
    input_dict = {"output_size": 7, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 10, 10).numpy()
    input_dict = {"output_size": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(3, 3, 15, 20).astype(np.float32)
    input_dict = {"output_size": 5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(4, 1, 7, 5).astype(np.float64)
    input_dict = {"output_size": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 32, 32).numpy()
    input_dict = {"output_size": 4, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(1, 7, 9).astype(np.float32)
    input_dict = {"output_size": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 10, 64, 33).numpy()
    input_dict = {"output_size": 8, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(1, 1, 100, 77).astype(np.float64)
    input_dict = {"output_size": 13, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(16, 3, 2, 2).numpy()
    input_dict = {"output_size": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 4, 13, 17).astype(np.float32)
    input_dict = {"output_size": 6, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 5, 9, 9).numpy()
    input_dict = {"output_size": 9, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(7, 3, 23, 19).astype(np.float64)
    input_dict = {"output_size": 11, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveAvgPool2d_1"] = adaptiveavgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveAvgPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool2d_1'.")


check_valid('torch.nn.AdaptiveAvgPool2d', generated_inputs['torch.nn.AdaptiveAvgPool2d_1'], lib="torch", suffix=1)
