
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def full_like_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, default values
    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "fill_value": 2.5,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor, specified dtype
    input2 = torch.randint(0, 10, (5, 5)).numpy()
    input_dict2 = {
        "input": input2,
        "fill_value": 5.0,
        "dtype": torch.float64,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Bool tensor, negative fill_value
    input3 = torch.tensor([[True, False], [False, True]]).numpy()
    input_dict3 = {
        "input": input3,
        "fill_value": -1.0,
        "dtype": torch.int32,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Complex tensor, specified layout
    input4 = torch.complex(torch.randn(3, 2), torch.randn(3, 2)).numpy()
    input_dict4 = {
        "input": input4,
        "fill_value": 1.0,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor, specified memory_format
    input5 = torch.arange(5).numpy()
    input_dict5 = {
        "input": input5,
        "fill_value": 10.0,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": torch.contiguous_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 4D tensor, different fill value
    input6 = torch.randn(1, 2, 3, 4).numpy()
    input_dict6 = {
        "input": input6,
        "fill_value": 0.0,
        "dtype": torch.float16,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.full_like"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.full_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.full_like'.")

check_valid('torch.full_like', generated_inputs['torch.full_like'], lib="torch")
