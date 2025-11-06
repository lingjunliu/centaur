
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def vsplit_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(16, dtype=torch.float32).reshape(4, 4).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.arange(-18, 0, dtype=torch.int64).reshape(6, 3).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.arange(32, dtype=torch.int16).reshape(8, 2, 2).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(12, 1, 2, 3, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (non-contiguous before numpy)
    input_arr = torch.arange(24, dtype=torch.float32).reshape(6, 4).t().contiguous().numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.zeros((10, 10), dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (boolean)
    input_arr = torch.tensor([[True, False, True, False, True],
                              [False, True, False, True, False]], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.linspace(-1, 1, steps=9, dtype=torch.float64).reshape(9, 1).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (zero-sized middle dimension)
    input_arr = torch.empty((16, 0, 7), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (higher-dimensional, sections=1)
    input_arr = torch.ones((2, 3, 4, 5, 6), dtype=torch.uint8).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (complex)
    input_arr = torch.randn(8, 2, dtype=torch.complex64).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.vsplit_1"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.vsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vsplit_1'.")


check_valid('torch.vsplit', generated_inputs['torch.vsplit_1'], lib="torch", suffix=1)
