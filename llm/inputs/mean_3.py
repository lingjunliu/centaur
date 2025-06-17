
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_mean_3_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, dim=1, keepdim=False
    input1 = torch.randn(4).numpy()
    dim1 = (0,)
    keepdim1 = False
    dtype1 = None
    out1 = None
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1, "dtype": dtype1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, dim=(0,1), keepdim=True
    input2 = torch.randn(2, 3).numpy()
    dim2 = (0, 1)
    keepdim2 = True
    dtype2 = None
    out2 = None
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2, "dtype": dtype2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Int tensor, dim=0, keepdim=False, specified dtype
    input3 = torch.randint(-5, 5, (3, 4)).numpy()
    dim3 = (0,)
    keepdim3 = False
    dtype3 = torch.float32
    out3 = None
    input_dict3 = {"input": input3, "dim": dim3, "keepdim": keepdim3, "dtype": dtype3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Complex tensor, dim=1, keepdim=True
    input4 = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).numpy()
    dim4 = (1,)
    keepdim4 = True
    dtype4 = None
    out4 = None
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4, "dtype": dtype4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor with negative values, multiple dimensions
    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = (1, 2)
    keepdim5 = False
    dtype5 = None
    out5 = None
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5, "dtype": dtype5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float tensor, dim is a single int, keepdim=True
    input6 = torch.randn(5, 5).numpy()
    dim6 = (0,)
    keepdim6 = True
    dtype6 = None
    out6 = None
    input_dict6 = {"input": input6, "dim": dim6, "keepdim": keepdim6, "dtype": dtype6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Float Tensor with all parameters
    input7 = torch.randn(2, 3, 4).numpy()
    dim7 = (0,)
    keepdim7 = True
    dtype7 = torch.float64
    out7 = torch.empty(1, 3, 4, dtype=torch.float64).numpy()
    input_dict7 = {"input": input7, "dim": dim7, "keepdim": keepdim7, "dtype": dtype7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.mean_3"] = torch_mean_3_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mean_3'.")

check_valid('torch.mean', generated_inputs['torch.mean_3'], lib="torch")
