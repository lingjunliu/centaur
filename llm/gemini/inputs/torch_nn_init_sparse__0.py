
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.zeros((5, 5), dtype=np.float32)
    sparsity = 0.1
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.zeros((10, 10), dtype=np.float64)
    sparsity = 0.5
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.zeros((20, 20), dtype=np.float16)
    sparsity = 0.9
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.zeros((30, 30), dtype=np.float32)
    sparsity = 0.2
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.zeros((40, 40), dtype=np.float64)
    sparsity = 0.7
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.zeros((100, 100), dtype=np.float32)
    sparsity = 0.01
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    tensor = np.zeros((2, 2), dtype=np.float32)
    sparsity = 0.3
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.zeros((12, 15), dtype=np.float32)
    sparsity = 0.8
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.zeros((25, 25), dtype=np.float32)
    sparsity = 0.4
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    tensor = np.zeros((50, 50), dtype=np.float64)
    sparsity = 0.6
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    tensor = np.zeros((2, 3), dtype=np.float32)
    sparsity = 0.99
    input_dict = {"tensor": tensor, "sparsity": sparsity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.sparse_"] = sparse_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.sparse_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.sparse_'.")

check_valid('torch.nn.init.sparse_', generated_inputs['torch.nn.init.sparse_'], lib="torch", suffix=0)
