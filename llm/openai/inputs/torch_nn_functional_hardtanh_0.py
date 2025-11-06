
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def hardtanh_inputs():
    list_of_inputs = []

    input_dict = {
        "input": torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32).numpy(),
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor([[-1.2, -0.3, 0.3],
                               [1.2, 2.5, -3.5]], dtype=torch.float64).numpy(),
        "min_val": -0.5,
        "max_val": 0.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor([[[-4, 0, 4],
                                [7, -8, 2]]], dtype=torch.int32).numpy(),
        "min_val": -3.0,
        "max_val": 3.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.arange(-24, 24, dtype=torch.float16).reshape(2, 3, 2, 4).numpy(),
        "min_val": -2.0,
        "max_val": 2.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor(0.0, dtype=torch.float32).numpy(),
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.arange(12, dtype=torch.float32).reshape(1, 2, 1, 2, 3).sub(6).numpy(),
        "min_val": 0.0,
        "max_val": 6.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor([0, 5, 10, 15, 25, 250], dtype=torch.uint8).numpy(),
        "min_val": 10.0,
        "max_val": 20.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor([-10.0, -5.5, -2.2, -0.1], dtype=torch.float64).numpy(),
        "min_val": -5.0,
        "max_val": -1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float32).numpy(),
        "min_val": 0.0,
        "max_val": 0.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor([float('inf'), float('-inf'), float('nan'), -0.0, 1.0], dtype=torch.float32).numpy(),
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.arange(-9, 9, dtype=torch.int64).reshape(3, 3, 2).numpy(),
        "min_val": -4.0,
        "max_val": 4.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.tensor([], dtype=torch.float32).numpy(),
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardtanh"] = hardtanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hardtanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardtanh'.")


check_valid('torch.nn.functional.hardtanh', generated_inputs['torch.nn.functional.hardtanh'], lib="torch", suffix=0)
