
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def torch_any_2_inputs():
    list_of_inputs = []
    
    input_arr = np.array([[True, False], [False, True], [True, True]])
    dim = 0
    keepdim = False
    out = np.empty(2, dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[True, True], [False, True], [True, True], [False, False]])
    dim = 1
    keepdim = False
    out = np.empty(4, dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[0, 1, 2], [0, 0, 0]])
    dim = 0
    keepdim = True
    out = np.empty((1, 3), dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[[1, 0], [0, 0]], [[0, 1], [1, 1]]])
    dim = 1
    keepdim = False
    out = np.empty((2, 2), dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 0]]])
    dim = 2
    keepdim = True
    out = np.empty((2, 2, 1), dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([0, 0, 0, 1, 0])
    dim = 0
    keepdim = False
    out = np.empty((), dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
    dim = 1
    keepdim = True
    out = np.empty((3, 1), dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[[0, 0, 0], [0, 0, 1]]])
    dim = 2
    keepdim = False
    out = np.empty((1, 2), dtype=bool)
    input_dict = {
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.any_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_2'.")


check_valid('torch.any', generated_inputs['torch.any_2'], lib="torch", suffix=2)
