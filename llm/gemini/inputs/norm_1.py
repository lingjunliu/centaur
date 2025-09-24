
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor with p=1 and dim - Corrected to float
    input_tensor = torch.randint(-5, 5, (2, 3, 2), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 1,
        "dim": (0, 1),
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor - Corrected dtype
    input_tensor = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 2,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": torch.complex64  # Explicitly setting the output type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with inf norm
    input_tensor = torch.randn(5).numpy()
    input_dict = {
        "input": input_tensor,
        "p": float('inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative inf norm
    input_tensor = torch.randn(5).numpy()
    input_dict = {
        "input": input_tensor,
        "p": float('-inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.norm_1"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_1'.")

check_valid('torch.norm', generated_inputs['torch.norm_1'], lib="torch")
