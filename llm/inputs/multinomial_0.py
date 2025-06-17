
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def multinomial_inputs():
    list_of_inputs = []

    # Input 1: Vector, no replacement
    input_tensor = np.array([0.1, 0.5, 0.2, 0.2], dtype=np.float32)
    num_samples = 2
    replacement = False
    input_dict = {
        "input": input_tensor,
        "num_samples": num_samples,
        "replacement": replacement,
        "generator": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix, with replacement
    input_tensor = np.array([[0.1, 0.5, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]], dtype=np.float64)
    num_samples = 5
    replacement = True
    input_dict = {
        "input": input_tensor,
        "num_samples": num_samples,
        "replacement": replacement,
        "generator": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector, with replacement, different num_samples
    input_tensor = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
    num_samples = 4
    replacement = True
    input_dict = {
        "input": input_tensor,
        "num_samples": num_samples,
        "replacement": replacement,
        "generator": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix, no replacement, different probabilities
    input_tensor = np.array([[0.9, 0.05, 0.03, 0.02], [0.2, 0.2, 0.3, 0.3]], dtype=np.float64)
    num_samples = 2
    replacement = False
    input_dict = {
        "input": input_tensor,
        "num_samples": num_samples,
        "replacement": replacement,
        "generator": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Vector, large number of samples, with replacement
    input_tensor = np.array([0.01, 0.9, 0.04, 0.05], dtype=np.float32)
    num_samples = 10
    replacement = True
    input_dict = {
        "input": input_tensor,
        "num_samples": num_samples,
        "replacement": replacement,
        "generator": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.multinomial"] = multinomial_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.multinomial' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.multinomial'.")

check_valid('torch.multinomial', generated_inputs['torch.multinomial'], lib="torch")
