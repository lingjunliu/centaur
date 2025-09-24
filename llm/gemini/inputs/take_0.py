
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_take_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D tensor and 1D index
    src = np.array([[4, 3, 5], [6, 7, 8]])
    index = np.array([0, 2, 5])
    input_dict = {"input": src, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 1D tensor and 1D index
    src = np.array([1, 2, 3, 4, 5])
    index = np.array([0, 1, 4])
    input_dict = {"input": src, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 3D tensor and 1D index
    src = np.arange(24).reshape(2, 3, 4)
    index = np.array([0, 5, 12, 23])
    input_dict = {"input": src, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Negative indices
    src = np.array([10, 20, 30, 40, 50])
    index = np.array([0, -1, -3])
    input_dict = {"input": src, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Different data type (float)
    src = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    index = np.array([0, 2, 4])
    input_dict = {"input": src, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Zero-dimensional input
    src = np.array(5)
    index = np.array([0])
    input_dict = {"input": src, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 7: Multidimensional Index
    src = np.array([[1, 2], [3, 4]])
    index = np.array([[0, 1], [2, 3]])
    input_dict = {"input": src, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.take"] = torch_take_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.take' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.take'.")

check_valid('torch.take', generated_inputs['torch.take'], lib="torch")
