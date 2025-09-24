
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sinh_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive values
    input_tensor = torch.tensor([0.5, 1.0, 1.5, 2.0]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor with negative values
    input_tensor = torch.tensor([-0.5, -1.0, -1.5, -2.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor with mixed values
    input_tensor = torch.tensor([[-0.5, 1.0], [-1.5, 2.0]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor with zeros
    input_tensor = torch.zeros((2, 2, 2)).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor with large values to potentially trigger Sleef behavior (CPU only)
    input_tensor = torch.tensor([10.0, -10.0, 20.0, -20.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor with a specified 'out' tensor
    input_tensor = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    out_tensor = torch.zeros((2, 2)).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor
    input_tensor = torch.tensor([]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sinh"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sinh'.")

check_valid('torch.sinh', generated_inputs['torch.sinh'], lib="torch", suffix=0)
