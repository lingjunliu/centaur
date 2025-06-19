
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logcumsumexp_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input1 = torch.randn(5).numpy()
    dim1 = 0
    out1 = torch.tensor([]).numpy()

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor
    input2 = torch.randn(2, 3).numpy()
    dim2 = 1
    out2 = torch.tensor([]).numpy()

    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    input3 = torch.randn(3, 4, 5).numpy()
    dim3 = 0
    out3 = torch.tensor([]).numpy()

    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Negative values
    input4 = torch.randn(4).numpy() - 2.0
    dim4 = 0
    out4 = torch.tensor([]).numpy()

    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different dimension
    input5 = torch.randn(2, 2).numpy()
    dim5 = 0
    out5 = torch.tensor([]).numpy()

    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logcumsumexp"] = logcumsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logcumsumexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logcumsumexp'.")

check_valid('torch.logcumsumexp', generated_inputs['torch.logcumsumexp'], lib="torch", suffix=0)
