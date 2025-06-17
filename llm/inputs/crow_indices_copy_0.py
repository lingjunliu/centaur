
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def crow_indices_copy_inputs():
    list_of_inputs = []

    # Example 1: Basic case with positive indices and row offsets
    indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    row_offsets = np.array([0, 2, 4, 6], dtype=np.int64)
    input_dict = {"indices": torch.tensor(indices), "row_offsets": torch.tensor(row_offsets)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.crow_indices_copy"] = crow_indices_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.crow_indices_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.crow_indices_copy'.")

check_valid('torch.crow_indices_copy', generated_inputs['torch.crow_indices_copy'], lib="torch")
