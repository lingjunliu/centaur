
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import torch
import copy

def nansum_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, np.nan, 4.0]),
        "dim": (),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0], [3.0, np.nan]]),
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
        "dim": (1,),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, np.nan]], [[5.0, 6.0], [np.nan, 8.0]]]),
        "dim": (0, 1),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([np.nan, np.nan, np.nan]),
        "dim": (),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "dim": (0,),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[[1.0, -2.0], [np.nan, 4.0]], [[-5.0, 6.0], [7.0, np.nan]]]]),
        "dim": (2,),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([np.nan]),
        "dim": (),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.5, 2.5, np.nan, -3.5], [np.nan, 5.5, 6.5, 7.5], [8.5, np.nan, 9.5, 10.5]]),
        "dim": (1,),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[-1.0, -2.0, np.nan], [-4.0, np.nan, -6.0]]),
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_3"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_3'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_3'], lib="torch", suffix=3)
