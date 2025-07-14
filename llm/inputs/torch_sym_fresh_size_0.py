
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np

def sym_fresh_size_inputs():
    list_of_inputs = []

    # Input 1: Basic example with a scalar
    input_dict = {"expr": "a"}
    list_of_inputs.append(input_dict)

    # Input 2: Another basic example
    input_dict = {"expr": "b"}
    list_of_inputs.append(input_dict)

    # Input 3: Yet another basic example
    input_dict = {"expr": "c"}
    list_of_inputs.append(input_dict)
    
    # Input 4: Still another basic example
    input_dict = {"expr": "d"}
    list_of_inputs.append(input_dict)

    # Input 5: One more basic example
    input_dict = {"expr": "e"}
    list_of_inputs.append(input_dict)

    # Input 6: Another example
    input_dict = {"expr": "f"}
    list_of_inputs.append(input_dict)

    # Input 7: Just another one
    input_dict = {"expr": "g"}
    list_of_inputs.append(input_dict)

    # Input 8: One more
    input_dict = {"expr": "h"}
    list_of_inputs.append(input_dict)

    # Input 9: A ninth one
    input_dict = {"expr": "i"}
    list_of_inputs.append(input_dict)

    # Input 10: A tenth one
    input_dict = {"expr": "j"}
    list_of_inputs.append(input_dict)
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sym_fresh_size"] = sym_fresh_size_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sym_fresh_size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_fresh_size'.")

check_valid('torch.sym_fresh_size', generated_inputs['torch.sym_fresh_size'], lib="torch", suffix=0)
