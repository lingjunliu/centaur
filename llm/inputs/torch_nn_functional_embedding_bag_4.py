
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(5, 3).numpy()
    weight_tensor = torch.randn(10, 3).numpy()
    indices_tensor = torch.tensor([0, 1, 2, 3, 4]).numpy()
    offsets_tensor = torch.tensor([0, 2, 4]).numpy()
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "sum"
    sparse = False
    per_sample_weights_tensor = torch.rand(5).numpy()
    include_last_offset = False
    
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(3, 5).numpy()
    weight_tensor = torch.randn(7, 5).numpy()
    indices_tensor = torch.tensor([0, 1, 2]).numpy()
    offsets_tensor = torch.tensor([0, 1]).numpy()
    max_norm = 2.0
    norm_type = 1.0
    scale_grad_by_freq = True
    mode = "mean"
    sparse = True
    per_sample_weights_tensor = torch.rand(3).numpy()
    include_last_offset = True

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_tensor = torch.randn(2, 2).numpy()
    weight_tensor = torch.randn(4, 2).numpy()
    indices_tensor = torch.tensor([0, 1, 2, 3]).numpy()
    offsets_tensor = torch.tensor([0, 1, 2, 3]).numpy()
    max_norm = float('nan') if np.isnan(np.nan) else np.nan
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "sum"
    sparse = False
    per_sample_weights_tensor = None
    include_last_offset = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(4, 1).numpy()
    weight_tensor = torch.randn(6, 1).numpy()
    indices_tensor = torch.tensor([0, 1, 2, 3, 4, 5]).numpy()
    offsets_tensor = torch.tensor([0, 3]).numpy()
    max_norm = 0.5
    norm_type = 0.5
    scale_grad_by_freq = True
    mode = "mean"
    sparse = True
    per_sample_weights_tensor = torch.rand(6).numpy()
    include_last_offset = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(1, 4).numpy()
    weight_tensor = torch.randn(2, 4).numpy()
    indices_tensor = torch.tensor([0, 1]).numpy()
    offsets_tensor = torch.tensor([0]).numpy()
    max_norm = 1.5
    norm_type = 3.0
    scale_grad_by_freq = False
    mode = "max"
    sparse = False
    per_sample_weights_tensor = torch.rand(2).numpy()
    include_last_offset = False

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "indices": indices_tensor,
        "offsets": offsets_tensor,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights_tensor,
        "include_last_offset": include_last_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding_bag_4"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_4'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_4'], lib="torch", suffix=4)
