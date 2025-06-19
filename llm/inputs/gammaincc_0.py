
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gammaincc_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different shapes and values
    input2 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    other2 = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Larger tensors
    input3 = np.random.rand(5, 5).astype(np.float32)
    other3 = np.random.rand(5, 5).astype(np.float32)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Scalar inputs
    input4 = np.array(2.5, dtype=np.float64)
    other4 = np.array(1.0, dtype=np.float64)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: One dimensional tensors
    input5 = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    other5 = np.array([2.0, 1.5, 1.0, 0.5], dtype=np.float32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.gammaincc"] = gammaincc_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.gammaincc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.gammaincc'.")

check_valid('torch.special.gammaincc', generated_inputs['torch.special.gammaincc'], lib="torch")
