
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(10, 3).numpy()
    weight = torch.randn(5, 3).numpy()
    indices = torch.tensor([0, 1, 2, 0, 3]).numpy()
    offsets = torch.tensor([0, 2, 3, 4]).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(5, 4).numpy()
    weight = torch.randn(3, 4).numpy()
    indices = torch.tensor([0, 1, 2, 1, 0, 2]).numpy()
    offsets = torch.tensor([0, 3]).numpy()
    per_sample_weights = torch.randn(6).numpy()

    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 1.0,
        "norm_type": 1.0,
        "scale_grad_by_freq": True,
        "mode": "mean",
        "sparse": True,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(8, 2).numpy()
    weight = torch.randn(4, 2).numpy()
    indices = torch.tensor([0, 1, 2, 3, 0, 1]).numpy()
    offsets = torch.tensor([0, 2, 4]).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randn(12, 5).numpy()
    weight = torch.randn(6, 5).numpy()
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]).numpy()
    offsets = torch.tensor([0, 4, 7]).numpy()
    per_sample_weights = torch.randn(11).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 2.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "mode": "sum",
        "sparse": True,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = torch.randn(3, 2).numpy()
    weight = torch.randn(2, 2).numpy()
    indices = torch.tensor([0, 1]).numpy()
    offsets = torch.tensor([0]).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - without max_norm
    input = torch.randn(3, 2).numpy()
    weight = torch.randn(2, 2).numpy()
    indices = torch.tensor([0, 1]).numpy()
    offsets = torch.tensor([0]).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding_bag_4"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_4'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_4'], lib="torch")
