
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor
    indices = np.array([[0, 1], [1, 2]]).astype(np.int64)
    values = np.array([1.0, 2.0]).astype(np.float32)
    size = (3, 4)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": torch.float32,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs["torch.sparse_coo_tensor_2"] = sparse_coo_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_coo_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_2'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_2'], lib="torch")
