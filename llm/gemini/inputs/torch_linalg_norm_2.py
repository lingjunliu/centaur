
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_norm_inputs():
    list_of_inputs = []

    # Input 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ord = 2.0
    dim = (0, 1)
    keepdim = False
    out = None
    dtype = torch.float32
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    A = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ord = 1.0
    dim = (0,)
    keepdim = True
    out = None
    dtype = torch.float64
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    A = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
    ord = -1.0
    dim = (0, 1)
    keepdim = False
    out = None
    dtype = torch.float32
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    A = torch.randn(2, 3, 4)
    ord = np.inf
    dim = (1, 2)
    keepdim = True
    out = None
    dtype = torch.float64
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    A = torch.randn(5)
    ord = -np.inf
    dim = (0,)
    keepdim = False
    out = None
    dtype = torch.float32
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ord = 'fro'
    dim = (0, 1)
    keepdim = True
    out = None
    dtype = torch.float64
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    A = torch.randn(2, 2, 2)
    ord = 2.0
    dim = (1,2)
    keepdim = False
    out = None
    dtype = torch.float32
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    A = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    ord = -2.0
    dim = (0,)
    keepdim = True
    out = None
    dtype = torch.float64
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    A = torch.randn(1, 5)
    ord = 2.0
    dim = (0,1)
    keepdim = False
    out = None
    dtype = torch.float32
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    A = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
    ord = -np.inf
    dim = (0, 1)
    keepdim = True
    out = None
    dtype = torch.float64
    input_dict = {"A": A.numpy(), "ord": ord, "dim": dim, "keepdim": keepdim, "out": out, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.norm_2"] = linalg_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.norm_2'.")

check_valid('torch.linalg.norm', generated_inputs['torch.linalg.norm_2'], lib="torch", suffix=2)
