
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def linalg_inv_inputs():
    list_of_inputs = []

    # Input 1: A simple 2x2 matrix
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A 3x3 matrix
    A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]).numpy()
    out = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).numpy()
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A batch of 2x2 matrices (batch size 2)
    A = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    out = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]).numpy()
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A matrix with negative values
    A = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: A larger matrix (4x4)
    A = torch.randn(4, 4).numpy()
    out = torch.randn(4, 4).numpy()
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Complex matrix
    A = torch.randn(2, 2, dtype=torch.complex128).numpy()
    out = torch.randn(2, 2, dtype=torch.complex128).numpy()
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.inv"] = linalg_inv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.inv'.")

check_valid('torch.linalg.inv', generated_inputs['torch.linalg.inv'], lib="torch", suffix=0)
