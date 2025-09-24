
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fft2_inputs():
    list_of_inputs = []

    # Input 1: Complex64 tensor, default s, dim, and norm
    input1 = torch.randn(10, 10, dtype=torch.complex64).numpy()
    input_dict1 = {"input": input1, "s": None, "dim": (-2, -1), "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float32 tensor, specified s and dim
    input2 = torch.randn(10, 10).numpy()
    s2 = (12, 12)
    dim2 = (0, 1)
    input_dict2 = {"input": input2, "s": s2, "dim": dim2, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Int64 tensor, specified norm
    input3 = torch.randint(-10, 10, (8, 8), dtype=torch.int64).numpy()
    norm3 = "forward"
    input_dict3 = {"input": input3, "s": None, "dim": (-2, -1), "norm": norm3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float64 tensor, specified s, dim, and norm
    input4 = torch.randn(16, 16, dtype=torch.float64).numpy()
    s4 = (8, 8)
    dim4 = (-1, -2)
    norm4 = "ortho"
    input_dict4 = {"input": input4, "s": s4, "dim": dim4, "norm": norm4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex128 tensor, with negative dimensions and s=-1
    input5 = torch.randn(5, 5, dtype=torch.complex128).numpy()
    s5 = (-1, -1)
    dim5 = (-2, -1)
    input_dict5 = {"input": input5, "s": s5, "dim": dim5, "norm": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.fft2"] = fft2_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.fft2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fft2'.")

check_valid('torch.fft.fft2', generated_inputs['torch.fft.fft2'], lib="torch")
