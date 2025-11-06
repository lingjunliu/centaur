
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def constantpad3d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {"padding": (1, 1, 2, 2, 3, 3), "value": 0.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 1, 2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {"padding": (0, 0, 0, 0, 0, 0), "value": 1.5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {"padding": (2, 0, 1, 3, 4, 0), "value": -2.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 2, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {"padding": (5, 0, 0, 5, 1, 2), "value": 3.5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {"padding": (0, 1, 0, 1, 0, 1), "value": 10.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 1, 10, 10, 10, dtype=torch.float32).numpy()
    input_dict = {"padding": (-2, -1, -3, -2, -1, -3), "value": 0.25, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 3, 1, 5, 1, dtype=torch.float32).numpy()
    input_dict = {"padding": (2, 3, 4, 0, 0, 1), "value": 2.2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 8, 4, 2, dtype=torch.float32).numpy()
    input_dict = {"padding": (0, 0, 2, 2, 1, 1), "value": -0.75, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 3, 3, dtype=torch.float16).numpy()
    input_dict = {"padding": (1, 2, 3, 4, 0, 0), "value": 5.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 1, 1, 1, dtype=torch.float32).numpy()
    input_dict = {"padding": (10, 9, 8, 7, 6, 5), "value": 0.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 2, 2, 2, 2, dtype=torch.float64).numpy()
    input_dict = {"padding": (0, 2, 1, 0, 3, 1), "value": -3.14159, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 5, 6, 7, dtype=torch.float32).numpy()
    input_dict = {"padding": (-1, 0, -2, -1, 0, -3), "value": 0.333, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad3d_2"] = constantpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad3d_2'.")


check_valid('torch.nn.ConstantPad3d', generated_inputs['torch.nn.ConstantPad3d_2'], lib="torch", suffix=2)
