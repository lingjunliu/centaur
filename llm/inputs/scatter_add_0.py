
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def scatter_add_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 1D tensors
    input = torch.zeros(5).long().numpy()
    index = torch.tensor([0, 2, 4, 2, 0]).numpy()
    src = torch.tensor([1, 2, 3, 4, 5]).long().numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensors with different shapes
    input = torch.zeros(3, 5).long().numpy()
    index = torch.tensor([[0, 1, 2, 0], [2, 0, 2, 1]]).numpy()
    src = torch.ones(2, 4).long().numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values in src
    input = torch.zeros(5).long().numpy()
    index = torch.tensor([0, 2, 4, 2, 0]).numpy()
    src = torch.tensor([-1, 2, -3, 4, -5]).long().numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different dim value
    input = torch.zeros(5, 3).long().numpy()
    index = torch.tensor([[0, 1, 2], [2, 0, 2], [1, 2, 0]]).numpy()
    src = torch.ones(3, 3).long().numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.scatter_add"] = scatter_add_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.scatter_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter_add'.")

check_valid('torch.scatter_add', generated_inputs['torch.scatter_add'], lib="torch")
