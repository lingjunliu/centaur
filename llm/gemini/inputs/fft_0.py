
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fft_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.arange(4).float().numpy()
    input_dict1 = {
        "input": input1,
        "n": None,
        "dim": -1,
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Complex tensor
    input2 = torch.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j]).numpy()
    input_dict2 = {
        "input": input2,
        "n": None,
        "dim": -1,
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor with specified n and dim
    input3 = torch.randn(8).float().numpy()
    input_dict3 = {
        "input": input3,
        "n": 4,
        "dim": 0,
        "norm": "forward",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Int tensor with norm = "ortho"
    input4 = torch.randint(0, 10, (16,)).int().numpy()
    input_dict4 = {
        "input": input4,
        "n": 32,
        "dim": 0,
        "norm": "ortho",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D float tensor
    input5 = torch.randn(4, 4).float().numpy()
    input_dict5 = {
        "input": input5,
        "n": None,
        "dim": 1,
        "norm": "backward",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D complex tensor
    input6 = (torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)).numpy()
    input_dict6 = {
        "input": input6,
        "n": 8,
        "dim": 2,
        "norm": "forward",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Negative values
    input7 = torch.arange(-5, 3).float().numpy()
    input_dict7 = {
        "input": input7,
        "n": None,
        "dim": 0,
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Long tensor
    input8 = torch.randint(0, 10, (4,)).long().numpy()
    input_dict8 = {
        "input": input8,
        "n": None,
        "dim": 0,
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.fft.fft"] = fft_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.fft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fft'.")

check_valid('torch.fft.fft', generated_inputs['torch.fft.fft'], lib="torch")
