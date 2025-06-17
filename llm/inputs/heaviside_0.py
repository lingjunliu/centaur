
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def heaviside_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensor and scalar value
    input_tensor = np.array([-1.5, 0, 2.0]).astype(np.float32)
    values_tensor = np.array([0.5]).astype(np.float32)
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Int tensor and int scalar
    input_tensor = np.array([-2, 0, 3]).astype(np.int32)
    values_tensor = np.array([2]).astype(np.int32)
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Scalar input and scalar value
    input_tensor = np.array(-1.0).astype(np.float32)
    values_tensor = np.array(0.5).astype(np.float32)
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Zero input and scalar value
    input_tensor = np.array(0.0).astype(np.float32)
    values_tensor = np.array(0.5).astype(np.float32)
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.heaviside"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.heaviside' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.heaviside'.")

check_valid('torch.heaviside', generated_inputs['torch.heaviside'], lib="torch")
