
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def chain_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    matrices = [torch.from_numpy(np.random.rand(2, 3).astype(np.float32)), torch.from_numpy(np.random.rand(3, 4).astype(np.float32)), torch.from_numpy(np.random.rand(4, 2).astype(np.float32))]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors
    matrices = [torch.from_numpy(np.random.randint(1, 5, size=(2, 3)).astype(np.int64)), torch.from_numpy(np.random.randint(1, 5, size=(3, 4)).astype(np.int64))]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrices with negative values
    matrices = [torch.from_numpy(np.random.randn(2, 3).astype(np.float32)), torch.from_numpy(np.random.randn(3, 4).astype(np.float32)), torch.from_numpy(np.random.randn(4, 2).astype(np.float32))]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex tensors
    matrices = [torch.from_numpy(np.random.rand(2, 3).astype(np.complex64) + 1j * np.random.rand(2, 3).astype(np.complex64)),
                torch.from_numpy(np.random.rand(3, 2).astype(np.complex64) + 1j * np.random.rand(3, 2).astype(np.complex64))]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More matrices with different shapes, ensuring matmul compatibility
    matrices = [torch.from_numpy(np.random.rand(5, 3).astype(np.float32)), torch.from_numpy(np.random.rand(3, 7).astype(np.float32)), torch.from_numpy(np.random.rand(7, 2).astype(np.float32))]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.chain_matmul"] = chain_matmul_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.chain_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chain_matmul'.")

check_valid('torch.chain_matmul', generated_inputs['torch.chain_matmul'], lib="torch")
