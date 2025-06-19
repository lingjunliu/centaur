
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1
    weight = torch.randn(10, 3).numpy()
    indices = torch.tensor([0, 1, 2, 3, 4]).numpy()
    offsets = torch.tensor([0, 3]).numpy()
    input = torch.randint(0, 10, (5,)).numpy()
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "sum"
    sparse = False
    per_sample_weights = torch.rand(5).numpy()
    include_last_offset = False

    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": np.float64(max_norm),
        "norm_type": np.float64(norm_type),
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": include_last_offset,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    weight = torch.randn(5, 4).numpy()
    indices = torch.tensor([0, 1, 2, 3, 4, 0, 1]).numpy()
    offsets = torch.tensor([0, 3, 7]).numpy()
    input = torch.randint(0, 5, (7,)).numpy()
    max_norm = 2.0
    norm_type = 1.0
    scale_grad_by_freq = True
    mode = "mean"
    sparse = True
    per_sample_weights = torch.rand(7).numpy()
    include_last_offset = False

    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": np.float64(max_norm),
        "norm_type": np.float64(norm_type),
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": include_last_offset,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    weight = torch.randn(12, 5).numpy()
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).numpy()
    offsets = torch.tensor([0, 2, 5, 10]).numpy()
    input = torch.randint(0, 12, (10,)).numpy()
    max_norm = 0.5
    norm_type = 0.5
    scale_grad_by_freq = False
    mode = "max"
    sparse = False
    per_sample_weights = torch.rand(10).numpy()
    include_last_offset = True
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": np.float64(max_norm),
        "norm_type": np.float64(norm_type),
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": include_last_offset,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    weight = torch.randn(8, 2).numpy()
    indices = torch.tensor([0, 1, 2, 3, 4, 5]).numpy()
    offsets = torch.tensor([0, 1, 3, 6]).numpy()
    input = torch.randint(0, 8, (6,)).numpy()
    max_norm = None
    norm_type = 2.0
    scale_grad_by_freq = True
    mode = "sum"
    sparse = True
    per_sample_weights = torch.rand(6).numpy()
    include_last_offset = False
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": max_norm,
        "norm_type": np.float64(norm_type),
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": include_last_offset,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    weight = torch.randn(3, 7).numpy()
    indices = torch.tensor([0, 1]).numpy()
    offsets = torch.tensor([0, 2]).numpy()
    input = torch.randint(0, 3, (2,)).numpy()
    max_norm = 3.0
    norm_type = 3.0
    scale_grad_by_freq = False
    mode = "mean"
    sparse = False
    per_sample_weights = torch.rand(2).numpy()
    include_last_offset = False
    input_dict = {
        "input": input,
        "weight": weight,
        "indices": indices,
        "offsets": offsets,
        "max_norm": np.float64(max_norm),
        "norm_type": np.float64(norm_type),
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "per_sample_weights": per_sample_weights,
        "include_last_offset": include_last_offset,
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
