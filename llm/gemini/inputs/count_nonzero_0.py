
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer tensor with some zeros
    input_tensor = np.array([0, 1, 2, 0, 3, 0], dtype=np.int64)
    input_dict = {"input": input_tensor, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float tensor with negative values and some zeros
    input_tensor = np.array([[0.0, -1.5, 2.0], [0.0, 0.0, 3.1]], dtype=np.float32)
    input_dict = {"input": input_tensor, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean tensor with some False values
    input_tensor = np.array([[[True, False], [True, True]], [[False, False], [True, False]]], dtype=np.bool_)
    input_dict = {"input": input_tensor, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimension (4D) integer tensor
    input_tensor = np.random.randint(-5, 5, size=(2, 2, 2, 2), dtype=np.int32)
    input_dict = {"input": input_tensor, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with all zeros
    input_tensor = np.zeros((3, 3), dtype=np.float64)
    input_dict = {"input": input_tensor, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.count_nonzero"] = count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.count_nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.count_nonzero'.")

check_valid('torch.count_nonzero', generated_inputs['torch.count_nonzero'], lib="torch")
