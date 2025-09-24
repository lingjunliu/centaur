
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def less_equal_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    other1 = 3.0
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Integer tensor with negative values
    input2 = np.array([-1, 0, 1, 2, 3])
    other2 = 1.0
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 2D float tensor
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other3 = 2.5
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Scalar input
    input4 = np.array(5.0)
    other4 = 5.0
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Test case 5: Tensor with all equal values
    input5 = np.array([2.0, 2.0, 2.0, 2.0])
    other5 = 2.0
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6:  3D integer tensor
    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other6 = 5.5
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: negative float values
    input7 = np.array([-1.0, -2.0, -3.0])
    other7 = -1.5
    input_dict7 = {"input": input7, "other": other7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.less_equal_2"] = less_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.less_equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.less_equal_2'.")

check_valid('torch.less_equal', generated_inputs['torch.less_equal_2'], lib="torch")
