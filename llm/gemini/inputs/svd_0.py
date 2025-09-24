
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def svd_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(5, 3).numpy()
    input_dict1 = {
        "input": input1,
        "some": True,
        "compute_uv": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Batch of matrices, some=False
    input2 = torch.randn(2, 4, 4).numpy()
    input_dict2 = {
        "input": input2,
        "some": False,
        "compute_uv": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: compute_uv=False
    input3 = torch.randn(3, 5).numpy()
    input_dict3 = {
        "input": input3,
        "some": True,
        "compute_uv": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Double tensor
    input4 = torch.randn(6, 2, dtype=torch.float64).numpy()
    input_dict4 = {
        "input": input4,
        "some": False,
        "compute_uv": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex tensor
    input5 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    input_dict5 = {
        "input": input5,
        "some": True,
        "compute_uv": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different dimensions
    input6 = torch.randn(1, 7, 3).numpy()
    input_dict6 = {
        "input": input6,
        "some": True,
        "compute_uv": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Rectangular matrix
    input7 = torch.randn(3, 7).numpy()
    input_dict7 = {
        "input": input7,
        "some": False,
        "compute_uv": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Zero matrix
    input8 = torch.zeros(5, 5).numpy()
    input_dict8 = {
        "input": input8,
        "some": True,
        "compute_uv": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.svd"] = svd_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.svd'.")

check_valid('torch.svd', generated_inputs['torch.svd'], lib="torch")
