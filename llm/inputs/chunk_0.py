
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_chunk_inputs():
    list_of_inputs = []

    # Input 1: 1D integer tensor, chunks divides dimension, dim=0
    input1 = torch.arange(12).numpy()
    chunks1 = 3
    dim1 = 0
    input_dict1 = {"input": input1, "chunks": chunks1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor, chunks does not divide dimension, dim=1
    input2 = torch.randn(5, 7).numpy()
    chunks2 = 4
    dim2 = 1
    input_dict2 = {"input": input2, "chunks": chunks2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex tensor, chunks > dimension size, dim=2
    input3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    chunks3 = 5
    dim3 = 2
    input_dict3 = {"input": input3, "chunks": chunks3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D integer tensor with negative values, dim=-1
    input4 = torch.randint(-10, 10, (2, 2, 2, 2)).numpy()
    chunks4 = 2
    dim4 = -1
    input_dict4 = {"input": input4, "chunks": chunks4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 5D float tensor, chunks divides dimension, dim=0
    input5 = torch.randn(4, 2, 2, 2, 2).numpy()
    chunks5 = 2
    dim5 = 0
    input_dict5 = {"input": input5, "chunks": chunks5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.chunk"] = torch_chunk_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.chunk' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chunk'.")

check_valid('torch.chunk', generated_inputs['torch.chunk'], lib="torch")
