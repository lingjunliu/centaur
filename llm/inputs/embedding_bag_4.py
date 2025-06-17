
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(10, 3).astype(np.float32),
        "weight": np.random.randn(5, 3).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 3, 4, 1, 2], dtype=np.int64),
        "offsets": np.array([0, 3, 6], dtype=np.int64),
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.random.randn(12, 4).astype(np.float32),
        "weight": np.random.randn(6, 4).astype(np.float32),
        "indices": np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int64),
        "offsets": np.array([0, 6], dtype=np.int64),
        "norm_type": 1.0,
        "scale_grad_by_freq": True,
        "mode": "mean",
        "sparse": True,
        "per_sample_weights": np.random.rand(12).astype(np.float32),
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.random.randn(8, 2).astype(np.float64),
        "weight": np.random.randn(4, 2).astype(np.float64),
        "indices": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        "offsets": np.array([0, 4], dtype=np.int64),
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.randn(15, 5).astype(np.float32),
        "weight": np.random.randn(7, 5).astype(np.float32),
        "indices": np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0], dtype=np.int64),
        "offsets": np.array([0, 7, 14], dtype=np.int64),
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.random.randn(5, 1).astype(np.float32),
        "weight": np.random.randn(3, 1).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 1], dtype=np.int64),
        "offsets": np.array([0, 3], dtype=np.int64),
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False
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
