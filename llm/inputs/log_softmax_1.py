
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input_1 = np.random.randn(3, 5).astype(np.float32)
    input_dict_1 = {"input": input_1, "dim": 1, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D float tensor with negative values
    input_2 = np.random.randn(2, 4, 6).astype(np.float64) - 1
    input_dict_2 = {"input": input_2, "dim": 2, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D integer tensor
    input_3 = np.array([1, 2, 3, 4, 5], dtype=np.int64).astype(np.float32)
    input_dict_3 = {"input": input_3, "dim": 0, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 4D float tensor
    input_4 = np.random.randn(1, 3, 7, 7).astype(np.float32)
    input_dict_4 = {"input": input_4, "dim": 3, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: 2D integer tensor with negative values
    input_5 = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32).astype(np.float32)
    input_dict_5 = {"input": input_5, "dim": 1, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.nn.functional.log_softmax_1"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.log_softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.log_softmax_1'.")

check_valid('torch.nn.functional.log_softmax', generated_inputs['torch.nn.functional.log_softmax_1'], lib="torch")
