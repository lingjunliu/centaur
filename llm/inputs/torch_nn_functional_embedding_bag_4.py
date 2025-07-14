
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    indices_tensor = np.array([0, 2, 1, 3], dtype=np.int64)
    offsets_tensor = np.array([0, 2], dtype=np.int64)
    max_norm_val = 1.0
    norm_type_val = 2.0
    scale_grad_by_freq_val = False
    mode_val = "sum"
    sparse_val = False
    per_sample_weights_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    include_last_offset_val = False
    
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    indices_tensor = np.array([0, 1], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = 2.0
    norm_type_val = 1.0
    scale_grad_by_freq_val = True
    mode_val = "mean"
    sparse_val = False
    per_sample_weights_val = None
    include_last_offset_val = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1.0, -2.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    indices_tensor = np.array([0, 1, 2], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = None
    norm_type_val = 2.0
    scale_grad_by_freq_val = False
    mode_val = "max"
    sparse_val = False
    per_sample_weights_val = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    include_last_offset_val = True

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    indices_tensor = np.array([0, 1], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = 1.5
    norm_type_val = 2.0
    scale_grad_by_freq_val = False
    mode_val = "sum"
    sparse_val = False
    per_sample_weights_val = np.array([0.2, 0.8], dtype=np.float32)
    include_last_offset_val = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1], [0.2]], dtype=np.float32)
    indices_tensor = np.array([0, 1], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = None
    norm_type_val = 2.0
    scale_grad_by_freq_val = True
    mode_val = "mean"
    sparse_val = False
    per_sample_weights_val = None
    include_last_offset_val = False
    
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], dtype=np.float32)
    indices_tensor = np.array([0, 1], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = 2.0
    norm_type_val = 1.5
    scale_grad_by_freq_val = False
    mode_val = "max"
    sparse_val = False
    per_sample_weights_val = np.array([1.0, 1.0], dtype=np.float32)
    include_last_offset_val = True
    
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[1.0, 2.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    indices_tensor = np.array([0, 1, 2], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = 0.5
    norm_type_val = 2.0
    scale_grad_by_freq_val = True
    mode_val = "sum"
    sparse_val = False
    per_sample_weights_val = np.array([0.3, 0.3, 0.4], dtype=np.float32)
    include_last_offset_val = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1]], dtype=np.float32)
    indices_tensor = np.array([0], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = 1.0
    norm_type_val = 2.0
    scale_grad_by_freq_val = False
    mode_val = "mean"
    sparse_val = False
    per_sample_weights_val = None
    include_last_offset_val = True

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.array([[1.0, 2.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    indices_tensor = np.array([0, 1, 2, 3], dtype=np.int64)
    offsets_tensor = np.array([0, 2], dtype=np.int64)
    max_norm_val = None
    norm_type_val = 1.0
    scale_grad_by_freq_val = True
    mode_val = "max"
    sparse_val = False
    per_sample_weights_val = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    include_last_offset_val = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    weight_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    indices_tensor = np.array([0, 1], dtype=np.int64)
    offsets_tensor = np.array([0], dtype=np.int64)
    max_norm_val = 1.0
    norm_type_val = 2.0
    scale_grad_by_freq_val = False
    mode_val = "sum"
    sparse_val = False
    per_sample_weights_val = np.array([0.5, 0.5], dtype=np.float32)
    include_last_offset_val = True

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm_val,
        "norm_type": norm_type_val,
        "scale_grad_by_freq": scale_grad_by_freq_val,
        "mode": mode_val,
        "sparse": sparse_val,
        "per_sample_weights": per_sample_weights_val,
        "include_last_offset": include_last_offset_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.embedding_bag_4"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_4'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_4'], lib="torch", suffix=4)
