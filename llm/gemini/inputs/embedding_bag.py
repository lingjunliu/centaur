
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Example 1: Basic case with sum mode
    input_dict = {
        'input': np.array([1, 2, 4, 5, 4, 3, 0], dtype=np.int64),
        'weight': np.random.rand(7, 3).astype(np.float32),
        'offsets': np.array([0, 4], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': False,
        'padding_idx': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: With per_sample_weights and sum mode
    input_dict = {
        'input': np.array([0, 1, 2, 3, 0], dtype=np.int64),
        'weight': np.random.rand(4, 5).astype(np.float32),
        'offsets': np.array([0, 2], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': np.array([0.5, 0.5, 0.5, 0.5, 0.5]).astype(np.float32),
        'include_last_offset': False,
        'padding_idx': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: With max mode and max_norm
    input_dict = {
        'input': np.array([0, 1, 2, 3], dtype=np.int64),
        'weight': np.random.rand(4, 4).astype(np.float32),
        'offsets': np.array([0], dtype=np.int64),
        'max_norm': 1.0,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'max',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': False,
        'padding_idx': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: With scale_grad_by_freq and padding_idx
    input_dict = {
        'input': np.array([0, 1, 2, 0], dtype=np.int64),
        'weight': np.random.rand(3, 2).astype(np.float32),
        'offsets': np.array([0, 2], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': True,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': False,
        'padding_idx': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: With include_last_offset
    input_dict = {
        'input': np.array([0, 1, 2, 3, 4, 0], dtype=np.int64),
        'weight': np.random.rand(5, 3).astype(np.float32),
        'offsets': np.array([0, 3, 6], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': True,
        'padding_idx': None
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
