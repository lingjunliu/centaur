
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []
    
    input_tensor = np.random.randn(3, 3)
    dim = 1
    keepdim = False
    out = np.empty(3)
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(4, 5)
    dim = 0
    keepdim = True
    out = np.empty((1, 5))
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4)
    dim = 2
    keepdim = False
    out = np.empty((2, 3))
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(10)
    dim = 0
    keepdim = False
    out = np.empty(())
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 5) - 2.0
    dim = 1
    keepdim = True
    out = np.empty((2, 1, 4, 5))
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(5, 6)
    dim = -1
    keepdim = False
    out = np.empty(5)
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 4) * 100
    dim = 1
    keepdim = False
    out = np.empty(3)
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 2)
    dim = 0
    keepdim = True
    out = np.empty((1, 2))
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(6, 7, 8)
    dim = 0
    keepdim = False
    out = np.empty((7, 8))
    input_dict = {
        "input": input_tensor,
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

if 'torch.logsumexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_1'.")


check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_1'], lib="torch", suffix=1)
