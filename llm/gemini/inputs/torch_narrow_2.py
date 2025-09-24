
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def narrow_inputs():
    list_of_inputs = []

    # Input 1, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    dim = 0
    start = torch.tensor(0).numpy()
    length = 2

    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    dim = 1
    start = torch.tensor(1).numpy()
    length = 2

    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, negative start
    input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    dim = -1
    start = torch.tensor(-1).numpy()
    length = 1

    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, 1D tensor
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    dim = 0
    start = torch.tensor(1).numpy()
    length = 3

    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, multi-dimensional
    input = torch.randn(2, 3, 4).numpy()
    dim = 1
    start = torch.tensor(0).numpy()
    length = 2

    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, start from end of tensor
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    dim = 0
    start = torch.tensor(-3).numpy()
    length = 2
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, zero length
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    dim = 0
    start = torch.tensor(2).numpy()
    length = 0
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, start as tensor
    input = torch.randn(3, 3).numpy()
    dim = 0
    start = torch.tensor(1).numpy()
    length = 1
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9, valid, 3D tensor, negative dim
    input = torch.randn(2, 3, 4).numpy()
    dim = -2
    start = torch.tensor(1).numpy()
    length = 1
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    dim = 0
    start = torch.tensor(1).numpy()
    length = 1

    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.narrow_2"] = narrow_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.narrow_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.narrow_2'.")

check_valid('torch.narrow', generated_inputs['torch.narrow_2'], lib="torch", suffix=2)
