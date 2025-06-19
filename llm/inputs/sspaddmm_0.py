
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def sspaddmm_inputs():
    list_of_inputs = []

    # Input 1
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([2.0, 3.0])
    shape = (2, 2)
    input = torch.sparse_coo_tensor(indices, values, shape).to_dense().numpy()
    indices1 = torch.tensor([[0, 0], [1, 1]])
    values1 = torch.tensor([1.0, 4.0])
    shape1 = (2, 2)
    mat1 = torch.sparse_coo_tensor(indices1, values1, shape1, dtype=torch.float32).to_dense().numpy()
    mat2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32).numpy()
    beta = 0.5
    alpha = 0.2
    out = np.array([])

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = torch.tensor([[0, 0]])
    values = torch.tensor([1.0])
    shape = (3, 3)
    input = torch.sparse_coo_tensor(indices, values, shape).to_dense().numpy()
    indices1 = torch.tensor([[1, 1]])
    values1 = torch.tensor([2.0])
    shape1 = (3, 3)
    mat1 = torch.sparse_coo_tensor(indices1, values1, shape1, dtype=torch.float32).to_dense().numpy()
    mat2 = torch.tensor([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]], dtype=torch.float32).numpy()
    beta = -1.0
    alpha = 2.0
    out = np.array([])

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = torch.tensor([[0, 2], [2, 0]])
    values = torch.tensor([-4.0, 5.0])
    shape = (3, 3)
    input = torch.sparse_coo_tensor(indices, values, shape).to_dense().numpy()
    indices1 = torch.tensor([[1, 2]])
    values1 = torch.tensor([3.0])
    shape1 = (3, 3)
    mat1 = torch.sparse_coo_tensor(indices1, values1, shape1, dtype=torch.float32).to_dense().numpy()
    mat2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32).numpy()
    beta = 0.0
    alpha = 1.0
    out = np.array([])

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = torch.tensor([[0, 0]])
    values = torch.tensor([6.0])
    shape = (2, 3)
    input = torch.sparse_coo_tensor(indices, values, shape).to_dense().numpy()
    indices1 = torch.tensor([[0, 1]])
    values1 = torch.tensor([7.0])
    shape1 = (2, 3)
    mat1 = torch.sparse_coo_tensor(indices1, values1, shape1, dtype=torch.float32).to_dense().numpy()
    mat2 = torch.tensor([[18.0], [19.0], [20.0]], dtype=torch.float32).numpy()
    beta = 1.0
    alpha = 0.0
    out = np.array([])

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = torch.tensor([[1, 1]])
    values = torch.tensor([8.0])
    shape = (4, 2)
    input = torch.sparse_coo_tensor(indices, values, shape).to_dense().numpy()
    indices1 = torch.tensor([[2, 0]])
    values1 = torch.tensor([9.0])
    shape1 = (4, 2)
    mat1 = torch.sparse_coo_tensor(indices1, values1, shape1, dtype=torch.float32).to_dense().numpy()
    mat2 = torch.tensor([[21.0, 22.0], [23.0, 24.0]], dtype=torch.float32).numpy()
    beta = 0.25
    alpha = -0.5
    out = np.array([])

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sspaddmm"] = sspaddmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sspaddmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sspaddmm'.")

check_valid('torch.sspaddmm', generated_inputs['torch.sspaddmm'], lib="torch", suffix=0)
