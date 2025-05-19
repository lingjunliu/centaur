
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1: Basic example with sum mode
    input_dict = {
        "input": np.array([1, 2, 4, 5, 4, 3, 0], dtype=np.int64),
        "weight": np.random.rand(7, 3).astype(np.float32),
        "offsets": np.array([0, 1, 2, 4, 5, 7], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with mean mode
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 5).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": True,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with max mode and padding_idx
    input_dict = {
        "input": np.array([0, 1, 2, 0, 3], dtype=np.int64),
        "weight": np.random.rand(4, 4).astype(np.float32),
        "offsets": np.array([0, 3, 5], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Example with include_last_offset=True
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 2).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Example with smaller embedding dimension
    input_dict = {
        "input": np.array([1, 0, 2, 1], dtype=np.int64),
        "weight": np.random.rand(3, 1).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Example with per_sample_weights
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 5).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": True,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Example with per_sample_weights and scale_grad_by_freq=False
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 5).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = embedding_bag_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('embedding_bag', generated_inputs)
