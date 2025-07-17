
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]], dtype=np.float32)
    indices_tensor = np.array([0, 2, 4, 1, 3, 5], dtype=np.int64)
    offsets_tensor = np.array([0, 3], dtype=np.int64)
    max_norm = None
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "sum"
    sparse = False
    per_sample_weights_tensor = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    include_last_offset = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices_tensor = np.array([0, 1, 2, 3], dtype=np.int64)
    offsets_tensor = np.array([0, 2, 4], dtype=np.int64)
    max_norm = None
    norm_type = 1.0
    scale_grad_by_freq = True
    mode = "mean"
    sparse = True
    per_sample_weights_tensor = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    include_last_offset = True

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1]], dtype=np.float32)
    indices_tensor = np.array([0], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm = None
    norm_type = 0.5
    scale_grad_by_freq = False
    mode = "max"
    sparse = False
    per_sample_weights_tensor = np.array([1.0], dtype=np.float32)
    include_last_offset = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
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
