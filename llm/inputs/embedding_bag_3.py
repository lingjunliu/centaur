
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def embedding_bag_inputs():
    list_of_inputs = []

    # Example 1: Basic example with SUM mode
    weight = np.random.randn(10, 3).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    input_dict = {
        "input": None,
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

    # Example 2: With MEAN mode and per_sample_weights
    weight = np.random.randn(5, 4).astype(np.float32)
    indices = np.array([0, 1, 2, 0, 3, 4], dtype=np.int64)
    offsets = np.array([0, 3], dtype=np.int64)
    per_sample_weights = np.array([0.5, 0.5, 0.5, 1, 0.2, 0.3], dtype=np.float32)
    input_dict = {
        "input": None,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: With MAX mode
    weight = np.random.randn(3, 5).astype(np.float32)
    indices = np.array([0, 1, 0, 2], dtype=np.int64)
    offsets = np.array([0, 2], dtype=np.int64)
    input_dict = {
        "input": None,
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

    # Example 5: With max_norm
    weight = np.random.randn(6, 3).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    offsets = np.array([0, 3], dtype=np.int64)
    input_dict = {
        "input": None,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.embedding_bag_3"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_3'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_3'], lib="torch")
