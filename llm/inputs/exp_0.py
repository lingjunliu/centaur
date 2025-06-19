
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, math
import numpy as np

def exp_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values, out specified
    input_tensor = torch.tensor([0.0, 1.0, 2.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, negative values, no out
    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, mixed values
    input_tensor = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).numpy()
    out_tensor = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, all zeros
    input_tensor = torch.zeros((2, 2, 2)).numpy()
    out_tensor = torch.tensor(np.zeros((2,2,2))).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor, some large values
    input_tensor = torch.tensor([10.0, 20.0, 30.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.exp"] = exp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.exp'.")

check_valid('torch.exp', generated_inputs['torch.exp'], lib="torch", suffix=0)
