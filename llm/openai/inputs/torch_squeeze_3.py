
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def squeeze_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3, 1).numpy()
    dim = (1, 3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 2
    input_arr = torch.tensor([7], dtype=torch.int32).numpy()
    dim = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 3
    input_arr = torch.zeros(1, 3, 1, 1, 4, dtype=torch.float64).numpy()
    dim = (0, 2, 3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 4
    input_arr = torch.ones(1, 1, 1, dtype=torch.int64).numpy()
    dim = (0, 1, 2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 5
    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    dim = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 6
    input_arr = torch.zeros(2, 1, 1, 5, dtype=torch.float32).numpy()
    dim = (-3, -2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 7
    input_arr = torch.arange(6, dtype=torch.int16).reshape(1, 2, 1, 3, 1).numpy()
    dim = (-1, 2, 0)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 8
    input_arr = torch.zeros(1, 1, 1, 1, dtype=torch.float16).numpy()
    dim = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 9
    input_arr = torch.randn(3, 1, 4, 1, 5, 1, dtype=torch.float32).numpy()
    dim = (3, 5)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 10
    input_arr = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).numpy()
    dim = (-3,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 11
    input_arr = torch.tensor([[[True], [False], [True], [False]]], dtype=torch.bool).numpy()
    dim = (0, 2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 12
    input_arr = torch.zeros(2, 1, 2, dtype=torch.complex64).numpy()
    dim = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    return list_of_inputs

generated_inputs["torch.squeeze_3"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.squeeze_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.squeeze_3'.")


check_valid('torch.squeeze', generated_inputs['torch.squeeze_3'], lib="torch", suffix=3)
