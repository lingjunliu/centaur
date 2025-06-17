
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def index_put_inputs():
    list_of_inputs = []

    # Case 1: Basic case with a single index
    input_tensor = np.zeros((5, 5), dtype=np.float32)
    indices = [np.array([0, 2, 4])]
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Multiple indices for a 2D tensor with shape matching values
    input_tensor = np.zeros((5, 5), dtype=np.float32)
    indices = [np.array([0, 1, 2]), np.array([3, 4, 0])]
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Accumulate is True
    input_tensor = np.ones((3, 3), dtype=np.float32)
    indices = [np.array([0, 1]), np.array([1, 2])]
    values = np.array([1.0, 2.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Correct values shape for selected indices and accumulate
    input_tensor = np.zeros((5, 5), dtype=np.float32)
    indices = [np.array([0, 2, 4]), np.array([1, 3, 0])]
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative indices
    input_tensor = np.zeros((5, 5), dtype=np.float32)
    indices = [np.array([0, 2, -1])]
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Integer tensor
    input_tensor = np.zeros((5, 5), dtype=np.int32)
    indices = [np.array([0, 2, 4])]
    values = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Boolean tensor
    input_tensor = np.zeros((5, 5), dtype=bool)
    indices = [np.array([0, 2, 4])]
    values = np.array([True, False, True], dtype=bool)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.index_put_2"] = index_put_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.index_put_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_put_2'.")

check_valid('torch.index_put', generated_inputs['torch.index_put_2'], lib="torch")
