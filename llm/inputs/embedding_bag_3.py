
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_dict = {
        "weight": np.random.rand(10, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        "offsets": np.array([0, 3], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using per_sample_weights
    input_dict = {
        "weight": np.random.rand(10, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        "offsets": np.array([0, 3], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": np.random.rand(6).astype(np.float32),
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using max_norm
    input_dict = {
        "weight": np.random.rand(10, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        "offsets": np.array([0, 3], dtype=np.int64),
        "max_norm": 1.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different mode (max)
    input_dict = {
        "weight": np.random.rand(10, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        "offsets": np.array([0, 3], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: scale_grad_by_freq = True
    input_dict = {
        "weight": np.random.rand(10, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        "offsets": np.array([0, 3], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: include_last_offset = True
    input_dict = {
        "weight": np.random.rand(10, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int64),
        "offsets": np.array([0, 3, 7], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: sparse = True
    input_dict = {
        "weight": np.random.rand(10, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        "offsets": np.array([0, 3], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": True,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding_bag_3"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_3'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_3'], lib="torch")
