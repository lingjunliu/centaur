
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def eq_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor comparison with a float
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other_float = 2.0
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor comparison with a float
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    other_float = 3.0
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor comparison with a float
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    other_float = 5.0
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    input_tensor = np.array([-1.0, -2.0, 0.0, 1.0], dtype=np.float32)
    other_float = -1.0
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zeros
    input_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    other_float = 0.0
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtypes (float16)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    other_float = 1.0
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtypes (float64)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    other_float = 3.0
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    input_tensor = np.array([1e10, 2e10, 3e10], dtype=np.float32)
    other_float = 2e10
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Small values
    input_tensor = np.array([1e-10, 2e-10, 3e-10], dtype=np.float32)
    other_float = 1e-10
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 2D example
    input_tensor = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    other_float = 1.5
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: check with nan
    input_tensor = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    other_float = np.nan
    out_tensor = np.array([], dtype=np.bool_)
    input_dict = {"input": input_tensor, "other": other_float, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.eq_2"] = eq_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.eq_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eq_2'.")

check_valid('torch.eq', generated_inputs['torch.eq_2'], lib="torch", suffix=2)
