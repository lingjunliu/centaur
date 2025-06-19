
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_sum_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, dim=1, keepdim=False
    input1 = torch.randn(4,).numpy()
    input_dict1 = {
        "input": input1,
        "dim": 0,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor, dim=0, keepdim=True, dtype=torch.float64
    input2 = torch.randint(-5, 5, (2, 3)).numpy()
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor, dim=(0,1), keepdim=False
    input3 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict3 = {
        "input": input3,
        "dim": (0, 1),
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor, dim=None, keepdim=False (default), dtype=torch.int32
    input4 = torch.randn(3, 4, 5).numpy()
    input_dict4 = {
        "input": input4,
        "dim": None,
        "keepdim": False,
        "dtype": torch.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Long tensor, dim=1, keepdim=True
    input5 = torch.randint(0, 10, (5, 4, 3), dtype=torch.int64).numpy()
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "keepdim": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float tensor, dim=(0,2), keepdim=True, dtype=torch.float32
    input6 = torch.randn(4, 5, 6).numpy()
    input_dict6 = {
        "input": input6,
        "dim": (0, 2),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Bool tensor, dim=0
    input7 = torch.randint(0, 2, (3,)).bool().numpy()
    input_dict7 = {
        "input": input7,
        "dim": 0,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.sum_2"] = torch_sum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sum_2'.")

check_valid('torch.sum', generated_inputs['torch.sum_2'], lib="torch")
