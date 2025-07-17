
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def smm_inputs():
    list_of_inputs = []

    # Input 1
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64).t()
    values = torch.tensor([1, 2])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64).t()
    values = torch.tensor([5, 6])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64).t()
    values = torch.tensor([1, 2, 3, 4])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger matrices
    indices = torch.tensor([[0, 0], [1, 2], [2, 1]], dtype=torch.int64).t()
    values = torch.tensor([1, 2, 3])
    size = torch.Size([3, 3])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrices
    indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64).t()
    values = torch.tensor([1, 2])
    size = torch.Size([2, 3])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different values
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64).t()
    values = torch.tensor([-1, 2])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[1, -2], [-3, 4]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: All zeros
    indices = torch.tensor([[], []], dtype=torch.int64)
    values = torch.tensor([])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large sparse matrix
    indices = torch.tensor([[0, 0], [9, 9]], dtype=torch.int64).t()
    values = torch.tensor([1, 1])
    size = torch.Size([10, 10])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.randn(10, 5, dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Another rectangular case
    indices = torch.tensor([[0, 0], [1, 1], [2,2]], dtype=torch.int64).t()
    values = torch.tensor([1, 2, 3])
    size = torch.Size([3, 4])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.randn(4, 2, dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float type
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64).t()
    values = torch.tensor([1.5, 2.5])
    size = torch.Size([2, 2])
    input_sparse = torch.sparse_coo_tensor(indices, values, size)
    mat = torch.tensor([[1.1, 2.2], [3.3, 4.4]], dtype=torch.float32).numpy()
    input_dict = {"input": input_sparse, "mat": mat}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.smm"] = smm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.smm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.smm'.")

check_valid('torch.smm', generated_inputs['torch.smm'], lib="torch", suffix=0)
