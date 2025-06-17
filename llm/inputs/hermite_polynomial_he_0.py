
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def hermite_polynomial_he_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, n=0
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    n = 0
    input_dict = {"x": x, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor, n=1
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    n = 1
    input_dict = {"x": x, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional float tensor, n=3
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    n = 3
    input_dict = {"x": x, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integer tensor, n=4
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int64)
    n = 4
    input_dict = {"x": x, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger n with float input
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    n = 5
    input_dict = {"x": x, "n": n}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.special.hermite_polynomial_he"] = hermite_polynomial_he_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.hermite_polynomial_he' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.hermite_polynomial_he'.")

check_valid('torch.special.hermite_polynomial_he', generated_inputs['torch.special.hermite_polynomial_he'], lib="torch")
