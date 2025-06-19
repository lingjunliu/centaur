
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def zeros_like_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (5, 5)).numpy()
    input_dict2 = {
        "input": input2,
        "dtype": torch.float32,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.zeros(1, 2, 3, 4, dtype=torch.int64).numpy()
    input_dict3 = {
        "input": input3,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict4 = {
        "input": input4,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.ones(3, dtype=torch.bool).numpy()
    input_dict5 = {
        "input": input5,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    input_dict6 = {
        "input": input6,
        "dtype": torch.int32,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(1, 3, 5, 5).numpy()
    input_dict7 = {
        "input": input7,
        "dtype": None,
        "layout": None,
        "requires_grad": True,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.zeros_like"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.zeros_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.zeros_like'.")

check_valid('torch.zeros_like', generated_inputs['torch.zeros_like'], lib="torch")
