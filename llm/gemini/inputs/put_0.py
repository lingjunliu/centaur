
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def put_inputs():
    list_of_inputs = []

    # Case 1: Basic test with float tensors
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([0, 2, 5, 11]).numpy()
    source_tensor = torch.randn(4).numpy()
    accumulate = False
    input_dict = {"input": input_tensor, "index": index_tensor, "source": source_tensor, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Int tensors, accumulate=True
    input_tensor = torch.randint(0, 10, (2, 5)).numpy()
    index_tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).numpy()
    source_tensor = torch.randint(0, 5, (10,)).numpy()
    accumulate = True
    input_dict = {"input": input_tensor, "index": index_tensor, "source": source_tensor, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensors
    input_tensor = torch.randn(10).numpy()
    index_tensor = torch.tensor([2, 4, 6, 8]).numpy()
    source_tensor = torch.randn(4).numpy()
    accumulate = False
    input_dict = {"input": input_tensor, "index": index_tensor, "source": source_tensor, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Multidimensional source and index
    input_tensor = torch.zeros(5, 5).numpy()
    index_tensor = torch.tensor([[0, 1], [2, 3]]).numpy()
    source_tensor = torch.ones(2, 2).numpy()
    accumulate = True
    input_dict = {"input": input_tensor, "index": index_tensor, "source": source_tensor, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values in source and index
    input_tensor = torch.randn(5, 5).numpy()
    index_tensor = torch.tensor([0, 2, 4, 6]).numpy()
    source_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    accumulate = False
    input_dict = {"input": input_tensor, "index": index_tensor, "source": source_tensor, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.put"] = put_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.put' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.put'.")

check_valid('torch.put', generated_inputs['torch.put'], lib="torch")
