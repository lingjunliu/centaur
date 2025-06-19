
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def nll_loss_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(3, 5, requires_grad=True).log_softmax(dim=1).detach().numpy()
    target = torch.tensor([1, 0, 4]).numpy()
    log_target = torch.tensor([]).numpy()
    weight = torch.tensor([0.2, 0.8, 0.3, 0.5, 0.1]).numpy()
    ignore_index = -100
    reduction = 'mean'
    input_dict = {
        'input': input,
        'target': target,
        'log_target': log_target,
        'weight': weight,
        'ignore_index': ignore_index,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(2, 3, 4, requires_grad=True).log_softmax(dim=1).detach().numpy()
    target = torch.tensor([[1, 0, 2, 1], [2, 1, 0, 0]]).numpy()
    log_target = torch.tensor([]).numpy()
    weight = torch.tensor([0.3, 0.7, 0.5]).numpy()
    ignore_index = -1
    reduction = 'none'
    input_dict = {
        'input': input,
        'target': target,
        'log_target': log_target,
        'weight': weight,
        'ignore_index': ignore_index,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(1, 4, requires_grad=True).log_softmax(dim=1).detach().numpy()
    target = torch.tensor([2]).numpy()
    log_target = torch.tensor([]).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    ignore_index = 2
    reduction = 'sum'
    input_dict = {
        'input': input,
        'target': target,
        'log_target': log_target,
        'weight': weight,
        'ignore_index': ignore_index,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randn(4, 2, requires_grad=True).log_softmax(dim=1).detach().numpy()
    target = torch.tensor([0, 1, 1, 0]).numpy()
    log_target = torch.tensor([]).numpy()
    weight = torch.tensor([0.6, 0.4]).numpy()
    ignore_index = -100
    reduction = 'mean'
    input_dict = {
        'input': input,
        'target': target,
        'log_target': log_target,
        'weight': weight,
        'ignore_index': ignore_index,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = torch.randn(2, 5, requires_grad=True).log_softmax(dim=1).detach().numpy()
    target = torch.tensor([3, 2]).numpy()
    log_target = torch.tensor([]).numpy()
    weight = None
    ignore_index = -100
    reduction = 'sum'
    input_dict = {
        'input': input,
        'target': target,
        'log_target': log_target,
        'weight': weight,
        'ignore_index': ignore_index,
        'reduction': reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.nll_loss_2"] = nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.nll_loss_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.nll_loss_2'.")

check_valid('torch.nn.functional.nll_loss', generated_inputs['torch.nn.functional.nll_loss_2'], lib="torch", suffix=2)
