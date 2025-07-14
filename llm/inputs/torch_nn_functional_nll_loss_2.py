
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def nll_loss_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([[-0.5, -0.2, -0.3], [-0.1, -0.8, -0.1]], dtype=np.float32)
    target = np.array([0, 2], dtype=np.int64)
    log_target = np.array([[-0.5, -0.2, -0.3], [-0.1, -0.8, -0.1]], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
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
    input = np.array([[-0.5, -0.2, -0.3], [-0.1, -0.8, -0.1]], dtype=np.float32)
    target = np.array([0, 2], dtype=np.int64)
    log_target = np.array([[-0.5, -0.2, -0.3], [-0.1, -0.8, -0.1]], dtype=np.float32)
    weight = np.array([0.2, 0.8, 0.5], dtype=np.float32)
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
    input = np.array([[-0.5, -0.2, -0.3]], dtype=np.float32)
    target = np.array([0], dtype=np.int64)
    log_target = np.array([[-0.5, -0.2, -0.3]], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ignore_index = 0
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
    input = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    target = np.array([1, 0], dtype=np.int64)
    log_target = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    weight = None
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
    input = np.array([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], dtype=np.float32)
    target = np.array([0, 1, 0], dtype=np.int64)
    log_target = np.array([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], dtype=np.float32)
    weight = np.array([0.5, 0.5], dtype=np.float32)
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

    # Input 6
    input = np.array([[-0.7, -0.3]], dtype=np.float32)
    target = np.array([1], dtype=np.int64)
    log_target = np.array([[-0.7, -0.3]], dtype=np.float32)
    weight = None
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

    # Input 7
    input = np.array([[-0.2, -0.8, -0.0]], dtype=np.float32)
    target = np.array([2], dtype=np.int64)
    log_target = np.array([[-0.2, -0.8, -0.0]], dtype=np.float32)
    weight = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ignore_index = 2
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

     # Input 8
    input = np.array([[-1.5, -0.5, -0.1], [-0.2, -1.0, -0.8]], dtype=np.float32)
    target = np.array([1, 2], dtype=np.int64)
    log_target = np.array([[-1.5, -0.5, -0.1], [-0.2, -1.0, -0.8]], dtype=np.float32)
    weight = np.array([0.3, 0.5, 0.2], dtype=np.float32)
    ignore_index = 1
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

    # Input 9
    input = np.array([[-2.0, -1.0], [-0.5, -0.5]], dtype=np.float32)
    target = np.array([0, 1], dtype=np.int64)
    log_target = np.array([[-2.0, -1.0], [-0.5, -0.5]], dtype=np.float32)
    weight = np.array([0.7, 0.3], dtype=np.float32)
    ignore_index = -100
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

    # Input 10
    input = np.array([[-0.9, -0.1, -0.0], [-0.4, -0.3, -0.3]], dtype=np.float32)
    target = np.array([0, 2], dtype=np.int64)
    log_target = np.array([[-0.9, -0.1, -0.0], [-0.4, -0.3, -0.3]], dtype=np.float32)
    weight = None
    ignore_index = 1
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

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.nll_loss_2"] = nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.nll_loss_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.nll_loss_2'.")

check_valid('torch.nn.functional.nll_loss', generated_inputs['torch.nn.functional.nll_loss_2'], lib="torch", suffix=2)
