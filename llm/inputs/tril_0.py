
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tril_inputs():
    list_of_inputs = []

    # Input 1: Basic case with diagonal=0
    input_tensor = torch.randn(3, 3).numpy()
    diagonal = 0
    out_tensor = torch.empty(0).numpy()
    input_dict = {"input": input_tensor, "diagonal": diagonal, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive diagonal
    input_tensor = torch.randn(4, 5).numpy()
    diagonal = 2
    out_tensor = torch.empty(0).numpy()
    input_dict = {"input": input_tensor, "diagonal": diagonal, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative diagonal
    input_tensor = torch.randn(5, 4).numpy()
    diagonal = -1
    out_tensor = torch.empty(0).numpy()
    input_dict = {"input": input_tensor, "diagonal": diagonal, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Square matrix with large diagonal
    input_tensor = torch.randn(6, 6).numpy()
    diagonal = 5
    out_tensor = torch.empty(0).numpy()
    input_dict = {"input": input_tensor, "diagonal": diagonal, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix with large negative diagonal
    input_tensor = torch.randn(7, 3).numpy()
    diagonal = -2
    out_tensor = torch.empty(0).numpy()
    input_dict = {"input": input_tensor, "diagonal": diagonal, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tril"] = tril_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tril'.")

check_valid('torch.tril', generated_inputs['torch.tril'], lib="torch", suffix=0)
