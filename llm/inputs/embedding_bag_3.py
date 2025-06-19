
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
    indices_tensor = torch.tensor([0, 1, 2, 0, 3]).numpy()
    offsets_tensor = torch.tensor([0, 3]).numpy()
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "sum"
    sparse = False
    per_sample_weights_tensor = torch.ones(5).numpy()
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
        "include_last_offset": include_last_offset,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding_bag_3"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding_bag_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_3'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_3'], lib="torch", suffix=3)
