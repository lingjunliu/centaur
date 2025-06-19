
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor, default p='fro', dim=None
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "p": 'fro', "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Integer tensor, p=2, dim=0 -> Convert to float
    input2 = torch.randint(-5, 5, (5, 5)).float().numpy()
    input_dict2 = {"input": input2, "p": 2, "dim": 0, "keepdim": True, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Complex tensor, p='fro', dim=(0, 1)
    input3 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "p": 'fro', "dim": (0, 1), "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Float tensor, p=float('inf'), dim=1, keepdim=True
    input4 = torch.randn(4, 2).numpy()
    input_dict4 = {"input": input4, "p": float('inf'), "dim": 1, "keepdim": True, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Float tensor, p=1, dim=(0, 2), dtype=torch.float64
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {"input": input5, "p": 1, "dim": (0, 2), "keepdim": False, "out": None, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Example 6: Negative float tensor, p=-2, dim=None
    input6 = (torch.randn(2, 3) - 2).numpy()
    input_dict6 = {"input": input6, "p": -2, "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Example 7: 1D Tensor, p = 1
    input7 = torch.arange(5, dtype=torch.float).numpy()
    input_dict7 = {"input": input7, "p": 1, "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.norm_2"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_2'.")

check_valid('torch.norm', generated_inputs['torch.norm_2'], lib="torch")
