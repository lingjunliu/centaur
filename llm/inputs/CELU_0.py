
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_1 = np.random.randn(2, 3).astype(np.float32)
    input_dict_1 = {
        "alpha": 1.0,
        "inplace": False,
        "input": input_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Float tensor with negative values
    input_2 = np.random.randn(4, 4).astype(np.float32)
    input_dict_2 = {
        "alpha": 0.5,
        "inplace": True,
        "input": input_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor
    input_3 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict_3 = {
        "alpha": 2.0,
        "inplace": False,
        "input": input_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar input (0-dimensional tensor)
    input_4 = np.array(-2.5).astype(np.float32)
    input_dict_4 = {
        "alpha": 1.5,
        "inplace": True,
        "input": input_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Tensor with larger values
    input_5 = np.random.randn(5, 5) * 10  # Scale up values
    input_5 = input_5.astype(np.float32)
    input_dict_5 = {
        "alpha": 1.0,
        "inplace": False,
        "input": input_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.nn.CELU"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.CELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CELU'.")

check_valid('torch.nn.CELU', generated_inputs['torch.nn.CELU'], lib="torch")
