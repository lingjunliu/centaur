
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def tensorinv_inputs():
    list_of_inputs = []

    # Input 1
    A = torch.eye(4 * 6).reshape((4, 6, 8, 3)).numpy()
    ind = np.int32(2)
    out = torch.tensor([]).numpy()

    input_dict = {
        "A": A,
        "ind": ind,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    A = torch.randn(4, 4).numpy()
    ind = np.int32(1)
    out = torch.tensor([]).numpy()

    input_dict = {
        "A": A,
        "ind": ind,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    A = torch.eye(2 * 3).reshape((2, 3, 2, 3)).numpy()
    ind = np.int32(2)
    out = torch.tensor([]).numpy()

    input_dict = {
        "A": A,
        "ind": ind,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    A = torch.randn(2, 2, 2, 2).numpy()
    A = A.reshape(4,4)
    ind = np.int32(1)
    out = torch.tensor([]).numpy()

    input_dict = {
        "A": A,
        "ind": ind,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    A = torch.randn(3, 3, 3, 3).numpy()
    A = A.reshape(9, 9)
    ind = np.int32(1)
    out = torch.tensor([]).numpy()

    input_dict = {
        "A": A,
        "ind": ind,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.tensorinv"] = tensorinv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.tensorinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.tensorinv'.")

check_valid('torch.linalg.tensorinv', generated_inputs['torch.linalg.tensorinv'], lib="torch", suffix=0)
