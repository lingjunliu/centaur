
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def rfftn_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(10, 10).numpy()
    s = (10, 10)
    dim = (0, 1)
    norm = "backward"
    out = torch.empty(10, 6, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(5, 5, 5).numpy()
    s = (5, 5, 5)
    dim = (0, 1, 2)
    norm = "forward"
    out = torch.empty(5, 5, 3, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(8, 8).numpy()
    s = (8, 8)
    dim = (0, 1)
    norm = "ortho"
    out = torch.empty(8, 5, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(16, 16, 16).numpy()
    s = (16, 16, 16)
    dim = (0, 1, 2)
    norm = "backward"
    out = torch.empty(16, 16, 9, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(4, 4, 4, 4).numpy()
    s = (4, 4, 4, 4)
    dim = (0, 1, 2, 3)
    norm = "forward"
    out = torch.empty(4, 4, 4, 3, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(32, 32).numpy()
    s = (32,)
    dim = (1,)
    norm = "ortho"
    out = torch.empty(32, 17, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(64, 64, 64).numpy()
    s = (64, 64, 64)
    dim = (0, 1, 2)
    norm = "backward"
    out = torch.empty(64, 64, 33, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(128, 128).numpy()
    s = (128, 128)
    dim = (0, 1)
    norm = "forward"
    out = torch.empty(128, 65, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.randn(256, 256).numpy()
    s = (256, 256)
    dim = (0, 1)
    norm = "ortho"
    out = torch.empty(256, 129, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(10, 10, 10).numpy()
    s = (10, 10, 10)
    dim = (0, 1, 2)
    norm = "backward"
    out = torch.empty(10, 10, 6, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.rfftn"] = rfftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.rfftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfftn'.")

check_valid('torch.fft.rfftn', generated_inputs['torch.fft.rfftn'], lib="torch", suffix=0)
