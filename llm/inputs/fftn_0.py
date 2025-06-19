
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def fftn_inputs():
    list_of_inputs = []

    # Case 1: Basic complex tensor, no s, dim, or norm
    x = torch.randn(10, 10, dtype=torch.complex64).numpy()
    input_dict = {"input": x, "s": None, "dim": None, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Real tensor, specifying s and dim
    x = torch.randn(10, 10, 10).numpy()
    s = (5, 5)
    dim = (0, 1)
    input_dict = {"input": x, "s": s, "dim": dim, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer tensor, specifying norm
    x = torch.randint(0, 10, (5, 5)).numpy()
    norm = "forward"
    input_dict = {"input": x, "s": None, "dim": None, "norm": norm, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D complex tensor with s, dim, and norm
    x = torch.randn(5, 5, 5, dtype=torch.complex128).numpy()
    s = (3, 3, 3)
    dim = (0, 1, 2)
    norm = "ortho"
    input_dict = {"input": x, "s": s, "dim": dim, "norm": norm, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Float tensor with negative values, s and dim
    x = torch.randn(8, 8).numpy() * -1
    s = (16, 16)
    dim = (0, 1)
    input_dict = {"input": x, "s": s, "dim": dim, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.fftn"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.fftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftn'.")

check_valid('torch.fft.fftn', generated_inputs['torch.fft.fftn'], lib="torch")
