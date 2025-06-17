
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def kthvalue_inputs():
    list_of_inputs = []

    # Test case 1: 1D tensor, default dim
    input1 = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6]).numpy()
    k1 = 3
    input_dict1 = {"input": input1, "k": k1, "dim": -1, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 2D tensor, dim=1, keepdim=True
    input2 = torch.tensor([[3, 1, 4], [1, 5, 9], [2, 6, 5]]).numpy()
    k2 = 2
    dim2 = 1
    keepdim2 = True
    input_dict2 = {"input": input2, "k": k2, "dim": dim2, "keepdim": keepdim2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 3D tensor, dim=0, keepdim=False
    input3 = torch.randn(3, 4, 5).numpy()
    k3 = 1
    dim3 = 0
    keepdim3 = False
    input_dict3 = {"input": input3, "k": k3, "dim": dim3, "keepdim": keepdim3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Negative values, k=1
    input4 = torch.tensor([-1, -5, 0, 2, -3]).numpy()
    k4 = 1
    input_dict4 = {"input": input4, "k": k4, "dim": -1, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: float tensor
    input5 = torch.randn(2, 3).numpy()
    k5 = 2
    input_dict5 = {"input": input5, "k": k5, "dim": 1, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.kthvalue"] = kthvalue_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.kthvalue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kthvalue'.")

check_valid('torch.kthvalue', generated_inputs['torch.kthvalue'], lib="torch")
