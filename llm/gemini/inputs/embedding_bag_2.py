
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1: Basic case with sum mode
    input_indices = np.array([1, 2, 4, 5, 4, 3, 0]).astype(np.int64)
    weight = np.random.rand(7, 3).astype(np.float32)
    offsets = np.array([0, 3]).astype(np.int64)
    input_dict = {
        "input": input_indices,
        "weight": weight,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Average mode WITHOUT per_sample_weights
    input_indices = np.array([1, 2, 4, 5, 4, 3, 0]).astype(np.int64)
    weight = np.random.rand(7, 3).astype(np.float32)
    offsets = np.array([0, 3]).astype(np.int64)
    input_dict = {
        "input": input_indices,
        "weight": weight,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Max mode with max_norm
    input_indices = np.array([1, 2, 4, 5, 4, 3, 0]).astype(np.int64)
    weight = np.random.rand(7, 3).astype(np.float32)
    offsets = np.array([0, 3]).astype(np.int64)
    input_dict = {
        "input": input_indices,
        "weight": weight,
        "offsets": offsets,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Include last offset
    input_indices = np.array([1, 2, 4, 5, 4, 3, 0]).astype(np.int64)
    weight = np.random.rand(7, 3).astype(np.float32)
    offsets = np.array([0, 3, 7]).astype(np.int64)
    input_dict = {
        "input": input_indices,
        "weight": weight,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different weight dimension
    input_indices = np.array([1, 2, 4, 5, 4, 3, 0]).astype(np.int64)
    weight = np.random.rand(7, 5).astype(np.float32)
    offsets = np.array([0, 3]).astype(np.int64)
    input_dict = {
        "input": input_indices,
        "weight": weight,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: scale_grad_by_freq = True
    input_indices = np.array([1, 2, 4, 5, 4, 3, 0]).astype(np.int64)
    weight = np.random.rand(7, 3).astype(np.float32)
    offsets = np.array([0, 3]).astype(np.int64)
    input_dict = {
        "input": input_indices,
        "weight": weight,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: per_sample_weights with mode = sum
    input_indices = np.array([1, 2, 4, 5, 4, 3, 0]).astype(np.int64)
    weight = np.random.rand(7, 3).astype(np.float32)
    offsets = np.array([0, 3]).astype(np.int64)
    per_sample_weights = np.random.rand(7).astype(np.float32)
    input_dict = {
        "input": input_indices,
        "weight": weight,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding_bag_2"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_2'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_2'], lib="torch")
