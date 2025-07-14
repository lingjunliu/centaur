
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_fft_ifft2_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(4, 4, dtype=torch.complex64).numpy()
    s = None
    dim = (-2, -1)
    norm = "backward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(8, 8, dtype=torch.complex64).numpy()
    s = (4, 4)
    dim = (-2, -1)
    norm = "ortho"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(16, 16, dtype=torch.complex64).numpy()
    s = (32, 32)
    dim = (-2, -1)
    norm = "forward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(2, 4, 8, 8, dtype=torch.complex64).numpy()
    s = (4, 4)
    dim = (-2, -1)
    norm = "backward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(2, 4, 8, 8, dtype=torch.complex64).numpy()
    s = (16, 16)
    dim = (-2, -1)
    norm = "ortho"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(3, 5, dtype=torch.complex64).numpy()
    s = (10, 10)
    dim = (-2, -1)
    norm = "forward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(5, 5, dtype=torch.complex64).numpy()
    s = (6, 6)
    dim = (-2, -1)
    norm = "backward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    s = (8, 8)
    dim = (-2, -1)
    norm = "ortho"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(4, 4, dtype=torch.complex64).numpy()
    s = (2, 2)
    dim = (-2, -1)
    norm = "forward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(8, 8, dtype=torch.complex64).numpy()
    s = (8, 8)
    dim = (-2, -1)
    norm = "backward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ifft2"] = torch_fft_ifft2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ifft2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifft2'.")

check_valid('torch.fft.ifft2', generated_inputs['torch.fft.ifft2'], lib="torch", suffix=0)
