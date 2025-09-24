
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pdist_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = 2.0
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    p = 1.0
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([[1.0, 2.0, 3.0, 4.0]])
    p = 0.5
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = np.inf
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    p = 3.0
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    p = 0.0
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.array([[1.0, 2.0, 3.0]])
    p = 2.0
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = 2.5
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = 1.5
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    p = 3.5
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    p = 0.1
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.pdist"] = pdist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.pdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pdist'.")

check_valid('torch.nn.functional.pdist', generated_inputs['torch.nn.functional.pdist'], lib="torch", suffix=0)
