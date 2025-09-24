
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def index_add_inputs():
    list_of_inputs = []

    # Case 1: Basic example with float tensor
    input_tensor = torch.zeros(5, 3, dtype=torch.float32).numpy()
    index_tensor = torch.tensor([0, 2, 4], dtype=torch.int64).numpy()
    source_tensor = torch.randn(3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different dimension, int tensor
    input_tensor = torch.ones(3, 4, 5, dtype=torch.int64).numpy()
    index_tensor = torch.tensor([0, 1], dtype=torch.int64).numpy()
    source_tensor = torch.randint(0, 10, (2, 4, 5), dtype=torch.int64).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D source
    input_tensor = torch.zeros(3, 3, dtype=torch.float32).numpy()
    index_tensor = torch.tensor([0, 1, 2], dtype=torch.int64).numpy()
    source_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: dim=1, adjusted sizes
    input_tensor = torch.randn(4, 5, dtype=torch.float32).numpy()
    index_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.int64).numpy()
    source_tensor = torch.randn(4, 5, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.index_add"] = index_add_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.index_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_add'.")

check_valid('torch.index_add', generated_inputs['torch.index_add'], lib="torch")
