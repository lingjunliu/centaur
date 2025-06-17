
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def index_put_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 1D indices and float values
    input_tensor = torch.randn(5, 5).numpy()
    indices = (np.array([0, 2, 4]),)
    values = torch.tensor([1.0, 2.0, 3.0]).float().numpy()
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D indices, integer values, accumulate=True
    input_tensor = torch.zeros(3, 3, dtype=torch.int64).numpy()
    indices = (np.array([0, 1]), np.array([1, 2]))
    values = torch.tensor([5, 6], dtype=torch.int64).numpy()
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shaped indices and values, float64
    input_tensor = torch.randn(4, 4, 4, dtype=torch.float64).numpy()
    indices = (np.array([0, 2]), np.array([1, 3]), np.array([0, 2]))
    values = torch.randn(2, dtype=torch.float64).numpy()
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative indices, boolean values
    input_tensor = torch.zeros(5, dtype=torch.bool).numpy()
    indices = (np.array([-1, -3]),)
    values = torch.tensor([True, True], dtype=torch.bool).numpy()
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Scalar value, accumulate=True
    input_tensor = torch.ones(2, 2, dtype=torch.float32).numpy()
    indices = (np.array([0]), np.array([0]))
    values = np.array(5.0, dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.index_put_1"] = index_put_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.index_put_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_put_1'.")

check_valid('torch.index_put', generated_inputs['torch.index_put_1'], lib="torch")
