
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_mean_inputs():
    list_of_inputs = []

    # Case 1: Float tensor, dim=0, keepdim=False
    input1 = torch.randn(2, 3, 4).numpy()
    dim1 = 0
    keepdim1 = False
    dtype1 = None
    out1 = None
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1, "dtype": dtype1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Int tensor, dim=1, keepdim=True, dtype=torch.float64
    input2 = torch.randint(-5, 5, (3, 5)).numpy()
    dim2 = 1
    keepdim2 = True
    dtype2 = torch.float64
    out2 = None
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2, "dtype": dtype2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Complex tensor, dim=(0, 1), keepdim=False
    input3 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input3 = input3.astype(np.complex64)
    dim3 = (0, 1)
    keepdim3 = False
    dtype3 = None
    out3 = None
    input_dict3 = {"input": input3, "dim": dim3, "keepdim": keepdim3, "dtype": dtype3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Float tensor, dim=2, keepdim=True, out tensor provided
    input4 = torch.randn(4, 4, 4).numpy()
    dim4 = 2
    keepdim4 = True
    dtype4 = None
    out4 = torch.empty(4, 4, 1).numpy()
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4, "dtype": dtype4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Float tensor, dim=-1, keepdim=False
    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = -1
    keepdim5 = False
    dtype5 = None
    out5 = None
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5, "dtype": dtype5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.mean_2"] = torch_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mean_2'.")

check_valid('torch.mean', generated_inputs['torch.mean_2'], lib="torch")
