
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def crow_indices_copy_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    indices = torch.tensor([0, 2, 1, 3], dtype=torch.int64).numpy()
    row_offsets = torch.tensor([0, 2, 4], dtype=torch.int64).numpy()

    input_dict = {
        "indices": indices,
        "row_offsets": row_offsets,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty indices
    indices = torch.tensor([], dtype=torch.int64).numpy()
    row_offsets = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()

    input_dict = {
        "indices": indices,
        "row_offsets": row_offsets,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single row
    indices = torch.tensor([0, 1, 2], dtype=torch.int64).numpy()
    row_offsets = torch.tensor([0, 3], dtype=torch.int64).numpy()

    input_dict = {
        "indices": indices,
        "row_offsets": row_offsets,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Irregular row lengths
    indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).numpy()
    row_offsets = torch.tensor([0, 1, 4, 6], dtype=torch.int64).numpy()

    input_dict = {
        "indices": indices,
        "row_offsets": row_offsets,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger indices
    indices = torch.tensor([100, 200, 150, 300], dtype=torch.int64).numpy()
    row_offsets = torch.tensor([0, 2, 4], dtype=torch.int64).numpy()

    input_dict = {
        "indices": indices,
        "row_offsets": row_offsets,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.crow_indices_copy"] = crow_indices_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.crow_indices_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.crow_indices_copy'.")

check_valid('torch.crow_indices_copy', generated_inputs['torch.crow_indices_copy'], lib="torch", suffix=0)
