
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bilinear_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.random.randn(2, 3).astype(np.float32)
    input2 = np.random.randn(2, 4).astype(np.float32)
    weight = np.random.randn(5, 3, 4).astype(np.float32)
    bias = np.random.randn(5).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different batch size
    input1 = np.random.randn(4, 5).astype(np.float32)
    input2 = np.random.randn(4, 2).astype(np.float32)
    weight = np.random.randn(3, 5, 2).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer tensors
    input1 = np.random.randint(-5, 5, size=(2, 3)).astype(np.int32)
    input2 = np.random.randint(-5, 5, size=(2, 4)).astype(np.int32)
    weight = np.random.randint(-5, 5, size=(5, 3, 4)).astype(np.int32)
    bias = np.random.randint(-5, 5, size=(5)).astype(np.int32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values
    input1 = np.random.randn(1, 2) * -1
    input2 = np.random.randn(1, 3) * -1
    weight = np.random.randn(4, 2, 3) * -1
    bias = np.random.randn(4) * -1
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Larger dimensions
    input1 = np.random.randn(3, 4).astype(np.float64)
    input2 = np.random.randn(3, 5).astype(np.float64)
    weight = np.random.randn(6, 4, 5).astype(np.float64)
    bias = np.random.randn(6).astype(np.float64)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.bilinear"] = bilinear_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bilinear'.")

check_valid('torch.bilinear', generated_inputs['torch.bilinear'], lib="torch")
