
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sym_fresh_size_inputs():
    list_of_inputs = []

    # Input 1: Simple case, symbolic expression
    input_dict = {"expr": torch.Size([torch.SymInt(1), torch.SymInt(2)])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Symbolic expression with 1 dimension
    input_dict = {"expr": torch.Size([torch.SymInt(5)])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Symbolic expression with multiple dimensions
    input_dict = {"expr": torch.Size([torch.SymInt(3), torch.SymInt(4), torch.SymInt(5)])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Symbolic expression with zero dimension
    input_dict = {"expr": torch.Size([torch.SymInt(0)])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Symbolic expression with mixed SymInt
    input_dict = {"expr": torch.Size([torch.SymInt(2), torch.SymInt(3)])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Symbolic expression with a large size
    input_dict = {"expr": torch.Size([torch.SymInt(1000)])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Symbolic expression with different sizes
    input_dict = {"expr": torch.Size([torch.SymInt(1), torch.SymInt(10), torch.SymInt(100)])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Symbolic expression with dimension 1
    input_dict = {"expr": torch.Size([torch.SymInt(1)])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Symbolic expression with dimension 2
    input_dict = {"expr": torch.Size([torch.SymInt(2)])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Symbolic expression with dimension 3
    input_dict = {"expr": torch.Size([torch.SymInt(3)])}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
