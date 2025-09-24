
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def addmv_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input_vec = torch.randn(2).numpy()
    mat = torch.randn(2, 3).numpy()
    vec = torch.randn(3).numpy()
    input_dict = {"input": input_vec, "mat": mat, "vec": vec, "beta": 1.0, "alpha": 1.0, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Integer tensors
    input_vec = torch.randint(0, 10, (3,)).numpy()
    mat = torch.randint(0, 10, (3, 4)).numpy()
    vec = torch.randint(0, 10, (4,)).numpy()
    input_dict = {"input": input_vec, "mat": mat, "vec": vec, "beta": 2, "alpha": 3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Negative values and beta=0
    input_vec = torch.randn(4).numpy()
    mat = torch.randn(4, 2).numpy()
    vec = torch.randn(2).numpy()
    input_dict = {"input": input_vec, "mat": mat, "vec": vec, "beta": 0.0, "alpha": -1.0, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Different alpha and beta values
    input_vec = torch.randn(5).numpy()
    mat = torch.randn(5, 5).numpy()
    vec = torch.randn(5).numpy()
    input_dict = {"input": input_vec, "mat": mat, "vec": vec, "beta": 0.5, "alpha": 2.5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Output tensor provided
    input_vec = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    out_tensor = torch.empty(3).numpy()
    input_dict = {"input": input_vec, "mat": mat, "vec": vec, "beta": 1.0, "alpha": 1.0, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Different shapes
    input_vec = torch.randn(10).numpy()
    mat = torch.randn(10, 7).numpy()
    vec = torch.randn(7).numpy()
    input_dict = {"input": input_vec, "mat": mat, "vec": vec, "beta": 0.75, "alpha": 1.25, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Large values
    input_vec = (torch.rand(4) * 100).numpy()
    mat = (torch.rand(4, 3) * 100).numpy()
    vec = (torch.rand(3) * 100).numpy()
    input_dict = {"input": input_vec, "mat": mat, "vec": vec, "beta": 2.0, "alpha": 0.5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.addmv"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addmv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv'.")

check_valid('torch.addmv', generated_inputs['torch.addmv'], lib="torch")
