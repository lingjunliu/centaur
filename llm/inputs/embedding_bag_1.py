
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def embedding_bag_inputs():
    list_of_inputs = []

    # Example 1: Basic case with sum mode
    input = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    weight = np.random.rand(10, 3).astype(np.float32)
    offsets = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

    input_dict = {
        "input": input,
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

    # Example 2: Average mode
    input = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    weight = np.random.rand(10, 3).astype(np.float32)
    offsets = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

    input_dict = {
        "input": input,
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

    # Example 3: Max mode with max_norm
    input = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    weight = np.random.rand(10, 3).astype(np.float32)
    offsets = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

    input_dict = {
        "input": input,
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

    # Example 4: Different offsets, sum mode
    input = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    weight = np.random.rand(10, 3).astype(np.float32)
    offsets = np.array([0, 2, 4, 6], dtype=np.int64)

    input_dict = {
        "input": input,
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

    # Example 5: scale_grad_by_freq = True
    input = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    weight = np.random.rand(10, 3).astype(np.float32)
    offsets = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

    input_dict = {
        "input": input,
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

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding_bag_1"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_1'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_1'], lib="torch")
