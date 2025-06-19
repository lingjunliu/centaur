
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def index_copy_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 2D float tensors
    input_tensor = torch.randn(3, 5).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    source_tensor = torch.randn(2, 5).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D int tensors with negative indices
    input_tensor = torch.randint(0, 10, (2, 4, 3)).numpy()
    index_tensor = torch.tensor([0, 1]).numpy()
    source_tensor = torch.randint(0, 10, (2, 4, 3)).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensors with different dimensions
    input_tensor = torch.arange(5, dtype=torch.float32).numpy()
    index_tensor = torch.tensor([1, 3]).numpy()
    source_tensor = torch.tensor([10.0, 20.0]).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multi-dimensional source tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    index_tensor = torch.tensor([0, 1]).numpy()
    source_tensor = torch.randn(2, 3, 4).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different dimension to copy along
    input_tensor = torch.randn(4, 5).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    source_tensor = torch.randn(2, 5).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.index_copy"] = index_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.index_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_copy'.")

check_valid('torch.index_copy', generated_inputs['torch.index_copy'], lib="torch")
