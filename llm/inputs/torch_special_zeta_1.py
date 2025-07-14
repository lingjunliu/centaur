
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def zeta_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    q = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1.5], dtype=np.float32)
    q = np.array([0.5], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    x = np.array([5.0, 6.0], dtype=np.float64)
    q = np.array([0.1, 0.2], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    x = np.array([2.0, 2.5, 3.0, 3.5], dtype=np.float32)
    q = np.array([1.0, 1.2, 1.4, 1.6], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([2.0], dtype=np.float64)
    q = np.array([1.0], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    q = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    x = np.array([1.0 + 1e-6, 2.0 + 1e-6], dtype=np.float32)
    q = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1.0 + 1e-6], dtype=np.float64)
    q = np.array([1.0], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    x = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    q = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    x = np.array(5, dtype=np.float32)
    q = np.array(2, dtype=np.float32)
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.zeta_1"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.zeta_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.zeta_1'.")

check_valid('torch.special.zeta', generated_inputs['torch.special.zeta_1'], lib="torch", suffix=1)
