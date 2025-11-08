
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def roll_inputs():
    list_of_inputs = []

    input_arr = torch.arange(10, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (3,), "dims": (0,)}))

    input_arr = torch.arange(1, 9, dtype=torch.int64).view(4, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (-1,), "dims": (0,)}))

    input_arr = torch.arange(15, dtype=torch.int32).view(3, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (2,), "dims": (1,)}))

    input_arr = torch.tensor([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [10, 11, 12]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (2, -1), "dims": (0, 1)}))

    input_arr = torch.arange(24, dtype=torch.float64).view(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (5,), "dims": (-1,)}))

    input_arr = torch.tensor([[[True, False, True]],
                              [[False, False, True]]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (1, -2), "dims": (0, 2)}))

    input_arr = torch.arange(3 * 4 * 5 * 6, dtype=torch.uint8).view(3, 4, 5, 6).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (-5, 7), "dims": (1, 3)}))

    input_arr = torch.randn(2, 2, 3, 1, 4, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (1, 2, -3), "dims": (0, 2, 4)}))

    input_arr = torch.tensor([42.0], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (12345,), "dims": (0,)}))

    input_arr = torch.arange(2 * 1 * 4, dtype=torch.float32).view(2, 1, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (3, -1), "dims": (1, 2)}))

    input_arr = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "shifts": (1, 3), "dims": (-4, -2)}))

    return list_of_inputs

generated_inputs["torch.roll_2"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.roll_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.roll_2'.")


check_valid('torch.roll', generated_inputs['torch.roll_2'], lib="torch", suffix=2)
