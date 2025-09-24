
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def modified_bessel_i1_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs["torch.special.modified_bessel_i1"] = modified_bessel_i1_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.modified_bessel_i1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.modified_bessel_i1'.")

check_valid('torch.special.modified_bessel_i1', generated_inputs['torch.special.modified_bessel_i1'], lib="torch")
