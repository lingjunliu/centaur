
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1: Basic case with reduction='mean'
    input_tensor = np.random.randn(3, 5).astype(np.float32)
    target_tensor = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: reduction='sum', with weights
    input_tensor = np.random.randn(2, 4, 3).astype(np.float64)
    target_tensor = np.random.randint(0, 2, size=(2, 4, 3)).astype(np.float64)
    weight_tensor = np.random.rand(2, 4, 3).astype(np.float64)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: reduction='none', with pos_weight
    input_tensor = np.random.randn(4, 2).astype(np.float32)
    target_tensor = np.random.randint(0, 2, size=(4, 2)).astype(np.float32)
    pos_weight_tensor = np.random.rand(2).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D input, reduction='mean', with weight
    input_tensor = np.random.randn(7).astype(np.float32)
    target_tensor = np.random.randint(0, 2, size=(7)).astype(np.float32)
    weight_tensor = np.random.rand(7).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional input, with all possible parameters (corrected pos_weight - matching dimension)
    input_tensor = np.random.randn(1, 3, 10, 10).astype(np.float64)
    target_tensor = np.random.randint(0, 2, size=(1, 3, 10, 10)).astype(np.float64)
    weight_tensor = np.random.rand(1, 3, 10, 10).astype(np.float64)
    pos_weight_tensor = np.random.rand(1, 3, 1, 1).astype(np.float64)

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.binary_cross_entropy_with_logits"] = binary_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.binary_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy_with_logits'.")

check_valid('torch.nn.functional.binary_cross_entropy_with_logits', generated_inputs['torch.nn.functional.binary_cross_entropy_with_logits'], lib="torch")
