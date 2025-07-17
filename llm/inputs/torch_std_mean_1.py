
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def std_mean_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    unbiased = True
    keepdim = False
    out = tuple()

    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    unbiased = False
    keepdim = True
    out = tuple()

    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    unbiased = True
    keepdim = True
    out = tuple()

    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    unbiased = False
    keepdim = False
    out = tuple()
    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    unbiased = True
    keepdim = False
    out = tuple()
    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([1.0], dtype=np.float32)
    unbiased = False
    keepdim = True
    out = tuple()
    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.array([1, 2, 3], dtype=np.int32)
    unbiased = True
    keepdim = False
    out = tuple()
    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.array([[1, 2], [3, 4]], dtype=np.int32)
    unbiased = False
    keepdim = True
    out = tuple()
    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    unbiased = True
    keepdim = False
    out = tuple()
    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    unbiased = False
    keepdim = True
    out = tuple()
    input_dict = {
        "input": torch.from_numpy(input).double(),
        "unbiased": unbiased,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.std_mean_1"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_1'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_1'], lib="torch", suffix=1)
