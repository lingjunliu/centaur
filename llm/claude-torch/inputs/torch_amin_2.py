
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import copy
import numpy as np

def amin_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[0.6451, -0.4866, 0.2987, -1.3312],
                             [-0.5744, 1.2980, 1.8397, -0.2713],
                             [0.9128, 0.9214, -1.7268, -0.2995],
                             [0.9023, 0.4853, 0.9075, -1.6165]])
    dim = (1,)
    keepdim = False
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 4, 5)
    dim = (0,)
    keepdim = False
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
    dim = (0,)
    keepdim = True
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4)
    dim = (0, 2)
    keepdim = False
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 5)
    dim = (2,)
    keepdim = True
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-5.0, -2.0, -8.0],
                             [-1.0, -3.0, -4.0]])
    dim = (1,)
    keepdim = False
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    dim = (0,)
    keepdim = False
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(5, 6, 7)
    dim = (1, 2)
    keepdim = True
    out = np.array([])
    input_dict = {
        "input": input_tensor,
        "dim": dim

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.amin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin_2'.")


check_valid('torch.amin', generated_inputs['torch.amin_2'], lib="torch", suffix=2)
