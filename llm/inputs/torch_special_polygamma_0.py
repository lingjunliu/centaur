
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def polygamma_inputs():
    list_of_inputs = []

    # Input 1: n=0, input=scalar
    n = 0
    input_tensor = np.array(1.0, dtype=np.float32)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: n=1, input=1D tensor
    n = 1
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: n=2, input=2D tensor
    n = 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: n=3, input=3D tensor
    n = 3
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: n=4, input with negative values
    n = 4
    input_tensor = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: n=5, large input values
    n = 5
    input_tensor = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: n=6, small input values
    n = 6
    input_tensor = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: n=7, input with zero values
    n = 7
    input_tensor = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: n=8, higher dimension input
    n = 8
    input_tensor = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: n=9, mixed positive and negative
    n = 9
    input_tensor = np.array([-1.0, 1.0, -2.0, 2.0], dtype=np.float64)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: n=10, zero dimension input
    n = 10
    input_tensor = np.array(1.0, dtype=np.float32)
    input_dict = {"n": n, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.polygamma"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.polygamma'.")

check_valid('torch.special.polygamma', generated_inputs['torch.special.polygamma'], lib="torch", suffix=0)
