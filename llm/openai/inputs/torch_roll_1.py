
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def roll_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "shifts": 1, "dims": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.linspace(0, 1, steps=5, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "shifts": -2, "dims": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.arange(12, dtype=torch.int64).view(3, 4).numpy()
    input_dict = {"input": input_arr, "shifts": 5, "dims": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.tensor([[1.5, -2.5, 3.0],
                              [4.2, 5.3, -6.1]], dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "shifts": 0, "dims": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.tensor([[[True, False],
                               [False, True]],
                              [[True, True],
                               [False, False]]], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "shifts": 1, "dims": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (torch.randn(3, 1, 4) + 1j * torch.randn(3, 1, 4)).to(torch.complex64).numpy()
    input_dict = {"input": input_arr, "shifts": -3, "dims": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.arange(2 * 2 * 3 * 4, dtype=torch.float16).view(2, 2, 3, 4).numpy()
    input_dict = {"input": input_arr, "shifts": 7, "dims": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.arange(1 * 3 * 4 * 5, dtype=torch.uint8).view(1, 3, 4, 5).numpy()
    input_dict = {"input": input_arr, "shifts": 13, "dims": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.arange(4, dtype=torch.float32).view(4, 1).numpy()
    input_dict = {"input": input_arr, "shifts": 1, "dims": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.arange(6, dtype=torch.int16).view(2, 3, 1).numpy()
    input_dict = {"input": input_arr, "shifts": -1, "dims": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.arange(12, dtype=torch.int8).view(1, 2, 1, 2, 3).numpy()
    input_dict = {"input": input_arr, "shifts": 4, "dims": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "shifts": 2, "dims": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    input_arr = torch.tensor([[10, 20, 30, 40],
                              [50, 60, 70, 80],
                              [90, 100, 110, 120]], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "shifts": -5, "dims": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14
    input_arr = torch.randn(5, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "shifts": 10, "dims": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15
    input_arr = torch.arange(2*3*4, dtype=torch.float32).view(2,3,4).numpy()
    input_dict = {"input": input_arr, "shifts": 3, "dims": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.roll_1"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.roll_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.roll_1'.")


check_valid('torch.roll', generated_inputs['torch.roll_1'], lib="torch", suffix=1)
