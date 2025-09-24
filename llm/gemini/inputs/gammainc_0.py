
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gammainc_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.random.rand(2, 3).astype(np.float32)
    other1 = np.random.rand(2, 3).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Different shapes, broadcastable
    input2 = np.random.rand(3).astype(np.float64)
    other2 = np.random.rand(2, 3).astype(np.float64)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Scalar input
    input3 = np.random.rand(1).astype(np.float32)[0]
    other3 = np.random.rand(2, 2).astype(np.float32)
    input_dict3 = {"input": np.array(input3), "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Higher dimensions
    input4 = np.random.rand(2, 3, 4).astype(np.float64)
    other4 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Integer type (promoted to float)
    input5 = np.random.randint(1, 10, size=(2, 2)).astype(np.int32)
    other5 = np.random.rand(2, 2).astype(np.float32)
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Case 6: Different dtypes that can be promoted
    input6 = np.random.rand(2, 3).astype(np.float16)
    other6 = np.random.rand(2, 3).astype(np.float32)
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.special.gammainc"] = gammainc_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.gammainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.gammainc'.")

check_valid('torch.special.gammainc', generated_inputs['torch.special.gammainc'], lib="torch")
