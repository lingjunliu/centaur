
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vecdot_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, default dim
    x = torch.randn(3, 2).numpy()
    y = torch.randn(3, 2).numpy()
    input_dict = {"x": x, "y": y, "dim": -1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors, dim=0 (Removed as it causes error)

    # Case 3: Different shapes but broadcastable, dim=1
    x = torch.randn(2, 3, 4).numpy()
    y = torch.randn(2, 3, 1).numpy()
    input_dict = {"x": x, "y": y, "dim": 1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Complex tensors, dim=-1
    x = torch.randn(5, 2, dtype=torch.complex64).numpy()
    y = torch.randn(5, 2, dtype=torch.complex64).numpy()
    input_dict = {"x": x, "y": y, "dim": -1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D tensors, dim=0
    x = torch.randn(5).numpy()
    y = torch.randn(5).numpy()
    input_dict = {"x": x, "y": y, "dim": 0, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Half tensors
    x = torch.randn(3, 2, dtype=torch.float16).numpy()
    y = torch.randn(3, 2, dtype=torch.float16).numpy()
    input_dict = {"x": x, "y": y, "dim": -1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.vecdot"] = vecdot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vecdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vecdot'.")

check_valid('torch.linalg.vecdot', generated_inputs['torch.linalg.vecdot'], lib="torch")
