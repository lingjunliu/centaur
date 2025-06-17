
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int tensors and 'ij' indexing
    x = torch.tensor([1, 2, 3]).numpy()
    y = torch.tensor([4, 5, 6]).numpy()
    input_dict = {
        "tensors": [x, y],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensors and 'xy' indexing
    xs = torch.linspace(-5, 5, steps=5).numpy()
    ys = torch.linspace(-5, 5, steps=5).numpy()
    input_dict = {
        "tensors": [xs, ys],
        "indexing": 'xy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Tensors with negative values
    x = torch.tensor([-1, 0, 1]).numpy()
    y = torch.tensor([4,5,6]).numpy()
    input_dict = {
        "tensors": [x, y],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three tensors
    x = torch.tensor([1, 2]).numpy()
    y = torch.tensor([3, 4]).numpy()
    z = torch.tensor([5, 6]).numpy()
    input_dict = {
        "tensors": [x, y, z],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single tensor
    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "tensors": [x],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Empty tensor
    x = np.array([])
    y = np.array([])
    input_dict = {
        "tensors": [x, y],
        "indexing": 'ij'
    }
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar input only
    # x = 5
    # input_dict = {
    #     "tensors": [x],
    #     "indexing": 'ij'
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs["torch.meshgrid"] = meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.meshgrid'.")

check_valid('torch.meshgrid', generated_inputs['torch.meshgrid'], lib="torch")
