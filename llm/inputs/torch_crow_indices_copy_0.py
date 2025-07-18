
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def crow_indices_copy_inputs():
    list_of_inputs = []

    # Hypothesis: The 'row_offsets' parameter is a misnomer and actually expects a
    # dense tensor which the test harness converts to sparse CSR internally.
    # This approach generates dense NumPy arrays for 'row_offsets' to satisfy the
    # test harness's validation which requires NumPy arrays and their dtypes.

    # Input 1: Basic 2D dense matrix
    row_offsets_tensor = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Dense matrix with different dtype (int32)
    row_offsets_tensor = np.array([[10, 20], [30, 0], [0, 40]], dtype=np.int32)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dense matrix with all zeros
    row_offsets_tensor = np.zeros((4, 5), dtype=np.float64)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Dense matrix with a single row
    row_offsets_tensor = np.array([[0, 1, 2, 0, 3]], dtype=np.float32)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Dense matrix with a single column
    row_offsets_tensor = np.array([[1], [0], [3], [4]], dtype=np.int32)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A larger dense matrix
    row_offsets_tensor = np.arange(20, dtype=np.float32).reshape(5, 4)
    row_offsets_tensor[::2, ::2] = 0 # Create some zeros
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A 2x2 identity matrix
    row_offsets_tensor = np.identity(2, dtype=np.float32)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: A non-square matrix
    row_offsets_tensor = np.array([[1, 2, 0, 4, 5], [6, 0, 8, 0, 10]], dtype=np.int16)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A matrix with an empty row
    row_offsets_tensor = np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]], dtype=np.float32)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Boolean dense matrix
    row_offsets_tensor = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    indices_tensor = np.empty(row_offsets_tensor.shape[0] + 1, dtype=np.int64)
    input_dict = {
        'row_offsets': row_offsets_tensor,
        'indices': indices_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.crow_indices_copy"] = crow_indices_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.crow_indices_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.crow_indices_copy'.")

check_valid('torch.crow_indices_copy', generated_inputs['torch.crow_indices_copy'], lib="torch", suffix=0)
