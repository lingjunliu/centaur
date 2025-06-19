
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_split_inputs():
    list_of_inputs = []

    a = torch.arange(10).reshape(5, 2).numpy()
    input_dict = {
        "tensor": a,
        "split_size_or_sections": 2,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.arange(10).reshape(5, 2).numpy()
    input_dict = {
        "tensor": a,
        "split_size_or_sections": 1,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.arange(10).reshape(2, 5).numpy()
    input_dict = {
        "tensor": a,
        "split_size_or_sections": 2,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(3, 4, 5).numpy()
    input_dict = {
        "tensor": a,
        "split_size_or_sections": 2,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "tensor": a,
        "split_size_or_sections": 2,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(5, 2).numpy()
    input_dict = {
        "tensor": a,
        "split_size_or_sections": 1,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "tensor": a,
        "split_size_or_sections": 1,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.split_1"] = torch_split_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.split_1'.")

check_valid('torch.split', generated_inputs['torch.split_1'], lib="torch")
