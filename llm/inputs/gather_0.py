
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_gather_inputs():
    list_of_inputs = []

    # Case 1: 2D tensor, dim=1
    input = np.array([[1, 2], [3, 4]])
    index = np.array([[0, 0], [1, 0]], dtype=np.int64)
    input_dict = {
        "input": input,
        "dim": 1,
        "index": index,
        "sparse_grad": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor, dim=0
    input = np.random.rand(2, 3, 4)
    index = np.random.randint(0, 2, size=(1, 3, 4), dtype=np.int64)
    input_dict = {
        "input": input,
        "dim": 0,
        "index": index,
        "sparse_grad": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor, dim=1
    input = np.random.rand(2, 3, 4)
    index = np.random.randint(0, 3, size=(2, 2, 4), dtype=np.int64)
    input_dict = {
        "input": input,
        "dim": 1,
        "index": index,
        "sparse_grad": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor, dim=2
    input = np.random.rand(2, 3, 4)
    index = np.random.randint(0, 4, size=(2, 3, 3), dtype=np.int64)
    input_dict = {
        "input": input,
        "dim": 2,
        "index": index,
        "sparse_grad": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: 2D tensor with negative indices, dim=0
    input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.int64)
    input_dict = {
        "input": input,
        "dim": 0,
        "index": index,
        "sparse_grad": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.gather"] = torch_gather_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gather'.")

check_valid('torch.gather', generated_inputs['torch.gather'], lib="torch")
