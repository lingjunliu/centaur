
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hspmm_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 2.0])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dimensions
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]])
    values = torch.tensor([1.0, 2.0, 3.0])
    size = (3, 3)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    out = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger matrix
    indices = torch.tensor([[0, 1, 2], [2, 3, 0]])
    values = torch.tensor([1.0, 2.0, 3.0])
    size = (4, 4)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.randn(4, 4).numpy()
    out = torch.zeros((4, 4)).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 2.0])
    size = (2, 3)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.randn(3, 4).numpy()
    out = torch.zeros((2, 4)).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([-1.0, 2.0])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, -2.0], [-3.0, 4.0]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: all zeros except one
    indices = torch.tensor([[0, 0]])
    values = torch.tensor([5.0])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: different data type
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1, 2])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    out = torch.tensor([[0, 0], [0, 0]]).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger size and different values
    indices = torch.tensor([[0, 1, 2], [2, 3, 0]])
    values = torch.tensor([4, 5, 6])
    size = (4, 4)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]).numpy()
    out = torch.zeros((4, 4), dtype=np.int64).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floats and ints mixed
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.5, 2.5])
    size = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"mat1": mat1, "mat2": mat2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More sparsity
    indices = torch.tensor([[0, 3], [2, 1]])
    values = torch.tensor([1.0, 2.0])
    size = (4, 4)
    mat1 = torch.sparse_coo_tensor(indices, values, size, requires_grad=False).to_dense().numpy()
    mat2 = torch.randn(4, 4).numpy()
    out = torch.zeros((4, 4)).numpy()
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
