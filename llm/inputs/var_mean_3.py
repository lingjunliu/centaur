
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def var_mean_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor, unbiased=True, keepdim=False
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dim = [0]
    unbiased = True
    keepdim = False
    out = ()
    input_dict = {"input": torch.from_numpy(input).double(), "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    
    input_dict["dim"] = tuple(input_dict["dim"])

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor, unbiased=False, keepdim=True
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dim = [0]
    unbiased = False
    keepdim = True
    out = ()
    input_dict = {"input": torch.from_numpy(input).double(), "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}

    input_dict["dim"] = tuple(input_dict["dim"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, unbiased=True, keepdim=False, multiple dims
    input = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dim = [0, 1]
    unbiased = True
    keepdim = False
    out = ()
    input_dict = {"input": torch.from_numpy(input).double(), "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}

    input_dict["dim"] = tuple(input_dict["dim"])
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with negative values, unbiased=False, keepdim=True, dim=None
    input = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    dim = [1]
    unbiased = False
    keepdim = True
    out = ()
    input_dict = {"input": torch.from_numpy(input).double(), "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}

    input_dict["dim"] = tuple(input_dict["dim"])
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor, unbiased=True, keepdim=True
    input = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dim = [0]
    unbiased = True
    keepdim = True
    out = ()
    input_dict = {"input": torch.from_numpy(input).double(), "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}

    input_dict["dim"] = tuple(input_dict["dim"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.var_mean_3"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_3'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_3'], lib="torch", suffix=3)
