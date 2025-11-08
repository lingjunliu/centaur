
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dropout3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(2, 3, 4, 8, 8).numpy()
    input_dict = {"p": 0.2, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(16, 4, 32, 32).numpy()
    input_dict = {"p": 0.5, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 1, 1, 2, 2).numpy()
    input_dict = {"p": 0.0, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(1, 1, 2, 2).numpy()
    input_dict = {"p": 1.0, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(2, 2, 3, 4, 5, dtype=torch.float64).numpy()
    input_dict = {"p": 0.33, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.randn(3, 2, 3, 3, 3, dtype=torch.float16).numpy()
    input_dict = {"p": 0.8, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    vals = torch.linspace(-1.0, 1.0, steps=32).reshape(2, 2, 2, 2, 2)
    input_arr = vals.numpy()
    input_dict = {"p": 0.1, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randn(1, 7, 3, 5, 4).numpy()
    input_dict = {"p": 0.25, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.arange(0, 3*4*4*4, dtype=torch.float32).reshape(3, 4, 4, 4).numpy()
    input_dict = {"p": 0.6, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.randn(2, 4, 1, 8, 8).numpy()
    input_dict = {"p": 0.4, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.randn(8, 3, 5, 2).numpy()
    input_dict = {"p": 0.05, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(2, 1, 10, 2, 2).numpy()
    input_dict = {"p": 0.7, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Dropout3d"] = dropout3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Dropout3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Dropout3d'.")


check_valid('torch.nn.Dropout3d', generated_inputs['torch.nn.Dropout3d'], lib="torch", suffix=0)
