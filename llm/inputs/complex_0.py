
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def complex_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D tensors
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).numpy()
    out = torch.empty(0, dtype=torch.complex64).numpy()
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D tensors
    real = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64).numpy()
    imag = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64).numpy()
    out = torch.empty(0, dtype=torch.complex128).numpy()
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 0D tensors (scalars)
    real = torch.tensor(1.0, dtype=torch.float32).numpy()
    imag = torch.tensor(2.0, dtype=torch.float32).numpy()
    out = torch.empty(0, dtype=torch.complex64).numpy()
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 3D tensors
    real = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    imag = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    out = torch.empty(0, dtype=torch.complex128).numpy()
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, negative values
    real = torch.tensor([-1.0, -2.0], dtype=torch.float32).numpy()
    imag = torch.tensor([-3.0, -4.0], dtype=torch.float32).numpy()
    out = torch.empty(0, dtype=torch.complex64).numpy()
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.complex"] = complex_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.complex'.")

check_valid('torch.complex', generated_inputs['torch.complex'], lib="torch", suffix=0)
