
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def where_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D boolean tensor
    condition = torch.tensor([True, False, True]).numpy()
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([4, 5, 6]).numpy()
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean tensor
    condition = torch.tensor([[True, False], [False, True]]).numpy()
    input_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    other_tensor = torch.tensor([[5, 6], [7, 8]]).numpy()
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean tensor
    condition = torch.tensor([[[True, False], [False, True]], [[False, True], [True, False]]]).numpy()
    input_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    other_tensor = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]).numpy()
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All True boolean tensor
    condition = torch.tensor([True, True, True]).numpy()
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([4, 5, 6]).numpy()
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All False boolean tensor
    condition = torch.tensor([False, False, False]).numpy()
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([4, 5, 6]).numpy()
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.where_2"] = where_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.where_2'.")

check_valid('torch.where', generated_inputs['torch.where_2'], lib="torch", suffix=2)
