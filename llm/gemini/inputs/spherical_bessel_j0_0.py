
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def spherical_bessel_j0_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, 1.5, 2.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.spherical_bessel_j0"] = spherical_bessel_j0_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.spherical_bessel_j0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.spherical_bessel_j0'.")

check_valid('torch.special.spherical_bessel_j0', generated_inputs['torch.special.spherical_bessel_j0'], lib="torch")
