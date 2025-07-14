
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D example
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    size = (3, 4)
    dtype = np.float32
    requires_grad = False
    input_dict = {"indices": indices, "values": values, "size": size, "dtype": np.dtype('float32'), "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sparse_coo_tensor_2"] = sparse_coo_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_coo_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_2'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_2'], lib="torch", suffix=2)
