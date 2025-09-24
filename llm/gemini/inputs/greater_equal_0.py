
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def greater_equal_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[2.0, 1.0], [3.0, 5.0]])
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[2, 1], [3, 5]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Scalar comparison
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other3 = 3.0
    input_dict3 = {"input": input3, "other": np.array(other3), "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Negative values
    input4 = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    other4 = np.array([[0.0, 1.0], [-2.0, 5.0]])
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Broadcasting with different shapes
    input5 = np.array([[1.0, 2.0, 3.0]])
    other5 = np.array([[2.0], [1.0]])
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.greater_equal"] = greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.greater_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.greater_equal'.")

check_valid('torch.greater_equal', generated_inputs['torch.greater_equal'], lib="torch")
