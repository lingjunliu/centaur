
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def median_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([3.0, 1.0, 4.0, 1.0, 5.0]),
        "dim": 0,
        "keepdim": False,
        "out": (np.array([]), np.array([], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "dim": 0,
        "keepdim": False,
        "out": (np.array([]), np.array([], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "dim": 1,
        "keepdim": False,
        "out": (np.array([]), np.array([], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[0.5, -0.5, 1.0], [-1.0, 0.0, 2.0]]),
        "dim": -1,
        "keepdim": True,
        "out": (np.array([[]]), np.array([[]], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "dim": 0,
        "keepdim": False,
        "out": (np.array([[]]), np.array([[]], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "dim": 1,
        "keepdim": True,
        "out": (np.array([[[]]]), np.array([[[]], [[]]], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "dim": 2,
        "keepdim": False,
        "out": (np.array([[]]), np.array([[]], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[-5.0, -3.0, -1.0], [-2.0, -4.0, -6.0]]),
        "dim": 0,
        "keepdim": False,
        "out": (np.array([]), np.array([], dtype=np.int64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[10.0, 20.0, 30.0, 40.0]]),
        "dim": 1,
        "keepdim": True,
        "out": (np.array([[]]), np.array([[

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.median_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.median_2'.")


check_valid('torch.median', generated_inputs['torch.median_2'], lib="torch", suffix=2)
