
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "p": 2.0, "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with p=1 and dim=0
    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {"input": input2, "p": 1.0, "dim": 0, "keepdim": True, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor with p='fro'
    input3 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "p": 'fro', "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with inf norm
    input4 = torch.arange(-5, 7, dtype=torch.float32).reshape(3, 4).numpy()
    input_dict4 = {"input": input4, "p": float('inf'), "dim": 1, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor with dim as a tuple
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {"input": input5, "p": 2.0, "dim": (1, 2), "keepdim": True, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Negative p value
    input6 = torch.randn(3, 4).numpy()
    input_dict6 = {"input": input6, "p": -2.0, "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Different dtype
    input7 = torch.randn(3, 4).numpy()
    input_dict7 = {"input": input7, "p": 2.0, "dim": None, "keepdim": False, "out": None, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.norm_9"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_9'.")

check_valid('torch.norm', generated_inputs['torch.norm_9'], lib="torch")
