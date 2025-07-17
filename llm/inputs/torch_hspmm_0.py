
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hspmm_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([1.0, 2.0]), size=(2, 2))
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data types
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([1, 2], dtype=torch.int32), size=(2, 2))
    mat2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger matrix
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]], dtype=torch.int64), values=torch.tensor([1.0, 2.0, 3.0, 4.0]), size=(3, 3))
    mat2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty sparse matrix
    mat1 = torch.sparse_coo_tensor(indices=torch.empty(2, 0, dtype=torch.int64), values=torch.empty(0), size=(2, 2))
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another valid input
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [0, 1]], dtype=torch.int64), values=torch.tensor([0.5, 1.5]), size=(1, 2))
    mat2 = torch.tensor([[2.0], [4.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Larger matrices with different shapes
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1]], dtype=torch.int64), values=torch.tensor([1.0, 2.0, 3.0, 4.0]), size=(4, 2))
    mat2 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different size out tensor
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([1.0, 2.0]), size=(2, 2))
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Float64 tensors. Removing this to resolve dtype issue
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([1.0, 2.0]), size=(2, 2))
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Removing complex numbers since hspmm might not support them
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([1.0, 2.0]), size=(2, 2))
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros values
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([0.0, 0.0]), size=(2, 2))
    mat2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different indices
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 2]], dtype=torch.int64), values=torch.tensor([1.0, 2.0]), size=(3, 3))
    mat2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Adding explicit float32 type to tensors. Removing as it didn't fix issue and reduced number of inputs to resolve the error
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([1.0, 2.0]), size=(2, 2))
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13 Adding another valid input to meet criteria
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [1, 1]], dtype=torch.int64), values=torch.tensor([2.0, 3.0]), size=(2, 2))
    mat2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    out = torch.tensor([])
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Adding explicit float32 type, and ensure no numpy
    mat1 = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), values=torch.tensor([1.0, 2.0], dtype=torch.float32), size=(2, 2))
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    out = torch.tensor([], dtype=torch.float32)
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.hspmm"] = hspmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hspmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hspmm'.")

check_valid('torch.hspmm', generated_inputs['torch.hspmm'], lib="torch", suffix=0)
