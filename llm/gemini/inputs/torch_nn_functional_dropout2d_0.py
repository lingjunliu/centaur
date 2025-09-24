
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dropout2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    p = 0.2
    training = True
    inplace = False
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 5, 20, 20).numpy()
    p = 0.8
    training = False
    inplace = False
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(4, 1, 5, 5).numpy()
    p = 0.5
    training = True
    inplace = True
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    p = 0.0
    training = True
    inplace = False
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    p = 1.0
    training = True
    inplace = False
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(8, 16, 32, 32).numpy()
    p = 0.3
    training = True
    inplace = False
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = torch.randn(1, 1, 2, 2).numpy()
    p = 0.7
    training = False
    inplace = False
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_tensor = torch.randn(3, 4, 7, 7).numpy()
    p = 0.1
    training = True
    inplace = False
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    p = 0.6
    training = True
    inplace = True
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = torch.randn(2, 2, 4, 4).numpy()
    p = 0.4
    training = False
    inplace = True
    input_dict = {"input": input_tensor, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.dropout2d"] = dropout2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.dropout2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout2d'.")

check_valid('torch.nn.functional.dropout2d', generated_inputs['torch.nn.functional.dropout2d'], lib="torch", suffix=0)
