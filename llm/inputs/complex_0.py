
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def complex_inputs():
    list_of_inputs = []

    # Case 1: float32 tensors
    real = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    imag = torch.tensor([4, 5, 6], dtype=torch.float32).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: float64 tensors, different shapes
    real = torch.randn(2, 2, dtype=torch.float64).numpy()
    imag = torch.randn(2, 2, dtype=torch.float64).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: float32 tensors, with negative values
    real = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float32).numpy()
    imag = torch.tensor([4.0, -5.0, 6.0], dtype=torch.float32).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.complex"] = complex_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.complex'.")

check_valid('torch.complex', generated_inputs['torch.complex'], lib="torch")
