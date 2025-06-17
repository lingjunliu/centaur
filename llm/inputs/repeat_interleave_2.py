
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def repeat_interleave_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor, scalar repeats
    input = np.array([1, 2, 3])
    repeats = np.array(2)
    dim = None
    out = None
    input_dict = {"input": input, "repeats": repeats, "dim": dim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor, 1D repeats, dim=0
    input = np.array([[1, 2], [3, 4]])
    repeats = np.array([1, 2])
    dim = 0
    out = None
    input_dict = {"input": input, "repeats": repeats, "dim": dim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensor, 1D repeats, dim=1
    input = np.array([[1, 2], [3, 4]])
    repeats = np.array([2, 1])
    dim = 1
    out = None
    input_dict = {"input": input, "repeats": repeats, "dim": dim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D tensor, list repeats
    input = np.array([1, 2, 3])
    repeats = np.array([1, 2, 3])
    dim = None
    out = None
    input_dict = {"input": input, "repeats": repeats, "dim": dim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.repeat_interleave_2"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.repeat_interleave_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_2'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_2'], lib="torch")
