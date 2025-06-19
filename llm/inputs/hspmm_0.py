
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
from scipy.sparse import random

def hspmm_inputs():
    list_of_inputs = []

    # Input 1
    mat1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.tensor([1.0, 2.0]),
        size=(2, 2)
    )
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.zeros((2,2))

    input_dict = {
        "mat1": mat1,
        "mat2": mat2,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    mat1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [1, 1]]),
        values=torch.tensor([3.0, 4.0]),
        size=(3, 2)
    )
    mat2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = torch.zeros((3,3))

    input_dict = {
        "mat1": mat1,
        "mat2": mat2,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    mat1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        values=torch.tensor([1.0, 2.0, 3.0]),
        size=(3, 3)
    )
    mat2 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    out = torch.zeros((3,3))

    input_dict = {
        "mat1": mat1,
        "mat2": mat2,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    mat1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.tensor([1.0, -2.0]),
        size=(1, 2)
    )
    mat2 = torch.tensor([[5.0], [6.0]])
    out = torch.zeros((1,1))

    input_dict = {
        "mat1": mat1,
        "mat2": mat2,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    mat1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [0]]),
        values=torch.tensor([2.0]),
        size=(1, 1)
    )
    mat2 = torch.tensor([[7.0]])
    out = torch.zeros((1,1))

    input_dict = {
        "mat1": mat1,
        "mat2": mat2,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.hspmm"] = hspmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hspmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hspmm'.")

check_valid('torch.hspmm', generated_inputs['torch.hspmm'], lib="torch", suffix=0)
