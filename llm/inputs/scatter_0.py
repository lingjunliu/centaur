
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_scatter_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with 2D tensors
    input = torch.randn(3, 4).numpy()
    dim = 1
    index = torch.tensor([[0, 1, 2, 0], [2, 0, 3, 0], [1, 0, 2, 3]]).long().numpy()
    src = torch.randn(3, 4).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 3D tensors
    input = torch.randn(2, 3, 4).numpy()
    dim = 0
    index = torch.tensor([[[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]]).long().numpy()
    src = torch.randn(1, 3, 4).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: src is a scalar
    input = torch.zeros(2, 5).numpy()
    dim = 1
    index = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]).long().numpy()
    src = 2.5
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Different data type (int)
    input = torch.zeros(3, 5, dtype=torch.int64).numpy()
    dim = 0
    index = torch.tensor([[0, 1, 2, 0, 1]]).long().numpy()
    src = torch.tensor([[1, 2, 3, 4, 5]]).long().numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Negative values in index
    input = torch.randn(3, 4).numpy()
    dim = 1
    index = torch.tensor([[0, 1, 2, 0], [2, 0, 3, 0], [1, 0, 2, 3]]).long().numpy()
    src = torch.randn(3, 4).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Larger tensors
    input = torch.randn(5, 6, 7).numpy()
    dim = 2
    index = torch.randint(0, 7, (5, 6, 7)).long().numpy()
    src = torch.randn(5, 6, 7).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: src with different shape for gather_add (scatter with add)
    input = torch.randn(2, 3).numpy()
    dim = 1
    index = torch.tensor([[0, 1, 2], [0, 1, 2]]).long().numpy()
    src = torch.randn(2, 3).numpy()
    input_dict = {"input": input, "dim": dim, "index": index, "src": src}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.scatter"] = torch_scatter_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter'.")

check_valid('torch.scatter', generated_inputs['torch.scatter'], lib="torch")
