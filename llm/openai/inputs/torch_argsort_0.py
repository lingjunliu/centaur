
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def argsort_inputs():
    list_of_inputs = []

    input = torch.tensor([3.0, -1.5, 2.2, 0.0, -3.3], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": 0, "descending": False, "stable": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[1.0, -2.0, 3.0],
                          [0.5, 0.5, -0.1]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": 1, "descending": True, "stable": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[2, 2, 1, 1],
                          [3, 3, 3, 3],
                          [0, 0, 0, 0]], dtype=torch.int64).numpy()
    input_dict = {"input": input, "dim": 0, "descending": False, "stable": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[1.0, 0.0, -1.0],
                           [2.0, 2.0, 2.0]],
                          [[float('nan'), 1.0, 1.0],
                           [float('inf'), -float('inf'), 0.0]]], dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim": -1, "descending": False, "stable": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(8, dtype=torch.int32).reshape(1, 4, 2).numpy()
    input_dict = {"input": input, "dim": 1, "descending": True, "stable": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([True, False, True, False, False], dtype=torch.bool).numpy()
    input_dict = {"input": input, "dim": 0, "descending": True, "stable": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((0,), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": 0, "descending": False, "stable": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((0, 3), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": 1, "descending": False, "stable": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    input_dict = {"input": input, "dim": 1, "descending": True, "stable": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[255, 0, 128],
                          [128, 128, 64]], dtype=torch.uint8).numpy()
    input_dict = {"input": input, "dim": 0, "descending": False, "stable": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": -2, "descending": True, "stable": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[5, 5],
                           [5, 5]],
                          [[5, 5],
                           [5, 4]]], dtype=torch.int64).numpy()
    input_dict = {"input": input, "dim": -1, "descending": False, "stable": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argsort'.")


check_valid('torch.argsort', generated_inputs['torch.argsort'], lib="torch", suffix=0)
