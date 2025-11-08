
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def AdaptiveMaxPool1d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(3, 8, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 5,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(2, 4, 16, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 1,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.tensor([[[-3.0]]], dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 1,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.linspace(-5, 5, steps=7, dtype=torch.float32).unsqueeze(0).numpy()
    input_dict = {
        "output_size": 7,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(3, 2, 5, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 10,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn(5, 1, 3, dtype=torch.float64).numpy()
    input_dict = {
        "output_size": 2,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.arange(80, dtype=torch.float32).reshape(4, 20).numpy()
    input_dict = {
        "output_size": 4,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn(1, 8, 50, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 25,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.tensor([[1.0, -2.0], [3.5, -4.5]], dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 3,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = torch.randn(2, 3, 9, dtype=torch.float16).numpy()
    input_dict = {
        "output_size": 4,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.randn(1, 5, 32, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 32,
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = torch.randn(7, 13, dtype=torch.float32).numpy()
    input_dict = {
        "output_size": 1,
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveMaxPool1d_1"] = AdaptiveMaxPool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveMaxPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool1d_1'.")


check_valid('torch.nn.AdaptiveMaxPool1d', generated_inputs['torch.nn.AdaptiveMaxPool1d_1'], lib="torch", suffix=1)
