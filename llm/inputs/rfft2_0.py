
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def rfft2_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    input_1 = np.random.rand(10, 10).astype(np.float32)
    input_dict_1 = {
        "input": input_1,
        "s": None,
        "dim": (-2, -1),
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: With specified signal size
    input_2 = np.random.rand(12, 15).astype(np.float64)
    input_dict_2 = {
        "input": input_2,
        "s": (8, 10),
        "dim": (-2, -1),
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With specified dimensions
    input_3 = np.random.rand(5, 6, 7, 8).astype(np.float32)
    input_dict_3 = {
        "input": input_3,
        "s": None,
        "dim": (-3, -2),
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With normalization
    input_4 = np.random.rand(7, 9).astype(np.float64)
    input_dict_4 = {
        "input": input_4,
        "s": None,
        "dim": (-2, -1),
        "norm": "forward",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: With negative values and specified signal size
    input_5 = np.random.randn(8, 11).astype(np.float32)
    input_dict_5 = {
        "input": input_5,
        "s": (16, -1),
        "dim": (-2, -1),
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.rfft2"] = rfft2_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.rfft2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfft2'.")

check_valid('torch.fft.rfft2', generated_inputs['torch.fft.rfft2'], lib="torch")
