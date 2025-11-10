
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def nansum_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([1., 2., float('nan'), 4.])
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "keepdim": False,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1., 2.], [3., float('nan')]])
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "keepdim": False,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1., 2.], [3., float('nan')]])
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "keepdim": False,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1., float('nan'), 3.], [4., 5., 6.]])
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "keepdim": True,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1., 2.], [3., 4.]], [[float('nan'), 6.], [7., 8.]]])
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "keepdim": False,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1., 2.], [3., float('nan')]], [[5., 6.], [7., 8.]]])
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "keepdim": True,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1., float('nan')], [3., 4.]], [[5., 6.], [float('nan'), 8.]]])
    input_dict = {
        "input": input_tensor,
        "dim": 2,
        "keepdim": False,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1., 2., float('nan')], [4., 5., 6.]])
    input_dict = {
        "input": input_tensor,
        "dim": -1,
        "keepdim": False,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1., 2.], [float('nan'), float('nan')]])
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "keepdim": True,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([5., float('nan'), 10., float('nan')])
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "keepdim": True,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_2"] = nan

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_2'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_2'], lib="torch", suffix=2)
