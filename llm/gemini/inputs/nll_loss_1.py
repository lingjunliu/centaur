
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def nll_loss_inputs():
    list_of_inputs = []

    # Example 1: Basic case with 1D target
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = np.array([1, 0, 4], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Different reduction method
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = np.array([1, 0, 4], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: With weight
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = np.array([1, 0, 4], dtype=np.int64)
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: With ignore_index
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = np.array([1, 0, -100], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: 2D target (batch_first=True)
    input = torch.randn(2, 5).log_softmax(dim=1).numpy()
    target = np.array([1, 0], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: None reduction
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = np.array([1, 0, 4], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: weight and ignore index
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = np.array([1, 0, -1], dtype=np.int64)
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": None,
        "ignore_index": -1,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.nll_loss_1"] = nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.nll_loss_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.nll_loss_1'.")

check_valid('torch.nn.functional.nll_loss', generated_inputs['torch.nn.functional.nll_loss_1'], lib="torch")
