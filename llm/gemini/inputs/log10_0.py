
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def log10_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with positive values
    input_tensor = torch.tensor([1.0, 10.0, 100.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with positive values
    input_tensor = torch.tensor([[0.1, 1.0], [10.0, 100.0]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with positive values
    input_tensor = torch.tensor([[[0.5, 5.0], [50.0, 500.0]], [[0.05, 0.5], [5.0, 50.0]]]).numpy()
    out_tensor = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with small positive values
    input_tensor = torch.tensor([0.001, 0.01, 0.1]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with a mix of values around 1
    input_tensor = torch.tensor([0.5, 1.0, 2.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty out tensor
    input_tensor = torch.tensor([2.0, 3.0, 4.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger tensor
    input_tensor = torch.rand(2, 3, 4).numpy() + 0.1
    out_tensor = torch.zeros(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.log10"] = log10_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log10'.")

check_valid('torch.log10', generated_inputs['torch.log10'], lib="torch", suffix=0)
