
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1
    weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    per_sample_weights = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    input_dict = {
        "input": np.array([0, 1, 2], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    weight = np.random.rand(5, 4).astype(np.float32)
    indices = np.array([0, 2, 1, 3, 4], dtype=np.int64)
    offsets = np.array([0, 2], dtype=np.int64)
    per_sample_weights = np.ones(5, dtype=np.float32)
    input_dict = {
        "input": np.array([0, 2, 1, 3, 4], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "mode": "mean",
        "sparse": True,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    weight = np.random.rand(3, 2).astype(np.float32)
    indices = np.array([0, 1, 0, 2], dtype=np.int64)
    offsets = np.array([0, 1, 3], dtype=np.int64)
    per_sample_weights = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    input_dict = {
        "input": np.array([0, 1, 0, 2], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 0.5,
        "norm_type": 1.0,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    weight = np.random.rand(4, 3).astype(np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    per_sample_weights = None
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": weight,
        "indices": indices,
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

     # Input 5
    weight = np.random.rand(2, 5).astype(np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    offsets = np.array([0, 1], dtype=np.int64)
    per_sample_weights = np.array([0.1, 0.9], dtype=np.float32)
    input_dict = {
        "input": np.array([0, 1], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "mode": "mean",
        "sparse": True,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    weight = np.random.rand(6, 2).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    per_sample_weights = None
    input_dict = {
        "input": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 0.7,
        "norm_type": 0.5,
        "scale_grad_by_freq": True,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    weight = np.random.rand(1, 3).astype(np.float32)
    indices = np.array([0], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    per_sample_weights = np.array([0.3], dtype=np.float32)
    input_dict = {
        "input": np.array([0], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 2.0,
        "norm_type": 3.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": True,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    weight = np.random.rand(4, 1).astype(np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    offsets = np.array([0, 2], dtype=np.int64)
    per_sample_weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": None,
        "norm_type": 1.5,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    weight = np.random.rand(3, 4).astype(np.float32)
    indices = np.array([0, 1, 2], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    per_sample_weights = None
    input_dict = {
        "input": np.array([0, 1, 2], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 0.3,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "mode": "max",
        "sparse": True,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    weight = np.random.rand(7, 3).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    offsets = np.array([0, 3], dtype=np.int64)
    per_sample_weights = np.random.rand(7).astype(np.float32)
    input_dict = {
        "input": np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": 1.5,
        "norm_type": 2.5,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.embedding_bag_3"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_3'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_3'], lib="torch", suffix=3)
