
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def diff_inputs():
    list_of_inputs = []

    # Case 1: Basic 1D tensor
    input = np.array([1, 3, 2])
    input_dict = {"input": input, "n": 1, "dim": -1, "prepend": None, "append": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 1D tensor with append
    input = np.array([1, 3, 2])
    append = np.array([4, 5])
    input_dict = {"input": input, "n": 1, "dim": -1, "prepend": None, "append": append, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensor with dim=0
    input = np.array([[1, 2, 3], [3, 4, 5]])
    input_dict = {"input": input, "n": 1, "dim": 0, "prepend": None, "append": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D tensor with dim=1
    input = np.array([[1, 2, 3], [3, 4, 5]])
    input_dict = {"input": input, "n": 1, "dim": 1, "prepend": None, "append": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Higher order difference (n=2)
    input = np.array([1, 3, 2, 4])
    input_dict = {"input": input, "n": 2, "dim": -1, "prepend": None, "append": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.diff"] = diff_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.diff' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diff'.")

check_valid('torch.diff', generated_inputs['torch.diff'], lib="torch")
