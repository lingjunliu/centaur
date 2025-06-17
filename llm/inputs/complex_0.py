
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def complex_inputs():
    list_of_inputs = []

    real = torch.tensor([1, 2], dtype=torch.float32)
    imag = torch.tensor([3, 4], dtype=torch.float32)
    input_dict = {
        "real": real.numpy(),
        "imag": imag.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.tensor([1.5, 2.5], dtype=torch.float64)
    imag = torch.tensor([3.5, 4.5], dtype=torch.float64)
    input_dict = {
        "real": real.numpy(),
        "imag": imag.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    imag = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    input_dict = {
        "real": real.numpy(),
        "imag": imag.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = torch.tensor([-1.0, -2.0], dtype=torch.float32)
    imag = torch.tensor([-3.0, -4.0], dtype=torch.float32)
    input_dict = {
        "real": real.numpy(),
        "imag": imag.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.tensor([1.0], dtype=torch.float32)
    imag = torch.tensor([3.0], dtype=torch.float32)
    input_dict = {
        "real": real.numpy(),
        "imag": imag.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.complex"] = complex_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.complex'.")

check_valid('torch.complex', generated_inputs['torch.complex'], lib="torch")
