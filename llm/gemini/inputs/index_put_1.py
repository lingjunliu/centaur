
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def index_put_inputs():
    list_of_inputs = []

    # Input 1: Basic example with 1D tensor
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    indices = (torch.tensor([0, 2, 4]).numpy(),)
    values = torch.tensor([10.0, 20.0, 30.0]).numpy()
    accumulate = False
    input_dict = {"input": input, "indices": indices, "values": values, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with boolean mask
    input = torch.zeros((3, 3)).numpy()
    indices = (torch.tensor([0, 1]).numpy(), torch.tensor([1, 2]).numpy())
    values = torch.ones((2,)).numpy()
    accumulate = False
    input_dict = {"input": input, "indices": indices, "values": values, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Accumulate is True
    input = torch.ones((3,)).numpy()
    indices = (torch.tensor([0, 1, 0]).numpy(),)
    values = torch.tensor([1.0, 2.0, 3.0]).numpy()
    accumulate = True
    input_dict = {"input": input, "indices": indices, "values": values, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional indices
    input = torch.zeros((2, 2, 2)).numpy()
    indices = (torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy())
    values = torch.ones((2,)).numpy()
    accumulate = False
    input_dict = {"input": input, "indices": indices, "values": values, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative indices
    input = torch.arange(5).float().numpy()
    indices = (torch.tensor([-1, -3]).numpy(),)
    values = torch.tensor([100.0, 200.0]).numpy()
    accumulate = False
    input_dict = {"input": input, "indices": indices, "values": values, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.index_put_1"] = index_put_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.index_put_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_put_1'.")

check_valid('torch.index_put', generated_inputs['torch.index_put_1'], lib="torch", suffix=1)
