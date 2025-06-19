
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def std_mean_inputs():
    list_of_inputs = []

    # Input 1, valid: 1D tensor, dim=0
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dim = 0
    unbiased = True
    keepdim = False
    out = (np.array([1.0]), np.array([1.0]))

    input_dict = {
        "input": input,
        "dim": dim,
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid: 2D tensor, dim=0
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = 0
    unbiased = False
    keepdim = True
    out = (np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid: 2D tensor, dim=1
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = 1
    unbiased = True
    keepdim = False
    out = (np.array([1.0, 1.0]), np.array([1.0, 1.0]))

    input_dict = {
        "input": input,
        "dim": dim,
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid: 3D tensor, dim=2
    input = torch.randn(2, 3, 4).numpy()
    dim = 2
    unbiased = False
    keepdim = True
    out = (np.zeros((2, 3, 1)), np.zeros((2, 3, 1)))

    input_dict = {
        "input": input,
        "dim": dim,
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid: negative values
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    dim = 0
    unbiased = True
    keepdim = False
    out = (np.array([1.0]), np.array([1.0]))

    input_dict = {
        "input": input,
        "dim": dim,
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.std_mean_3"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_3'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_3'], lib="torch", suffix=3)
