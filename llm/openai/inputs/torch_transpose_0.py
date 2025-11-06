
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def transpose_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(2, 3, dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(3, 2, dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim0": 1, "dim1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim0": -1, "dim1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randint(-10, 10, (2, 3, 4), dtype=torch.int64).numpy()
    input_dict = {"input": input, "dim0": 1, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.randn(4, 5, 6, 7, dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim0": -1, "dim1": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.randint(-100, 100, (4, 4), dtype=torch.int32).numpy()
    input_dict = {"input": input, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = torch.ones((2, 2, 2), dtype=torch.bool).numpy()
    input_dict = {"input": input, "dim0": 0, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.empty((0, 5), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = torch.randn(3, 3, dtype=torch.complex64).numpy()
    input_dict = {"input": input, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.randn(3, 1, 5, dtype=torch.float16).numpy()
    input_dict = {"input": input, "dim0": 0, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input = torch.randint(-128, 127, (1, 2, 3, 4), dtype=torch.int8).numpy()
    input_dict = {"input": input, "dim0": 2, "dim1": -4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input = torch.randn(5, 1, dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim0": -2, "dim1": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.transpose"] = transpose_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.transpose'.")


check_valid('torch.transpose', generated_inputs['torch.transpose'], lib="torch", suffix=0)
