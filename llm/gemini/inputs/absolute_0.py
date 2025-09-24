
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def absolute_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float tensor with negative values
    input1 = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    input_dict1 = {"input": input1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D integer tensor with a mix of positive and negative values
    input2 = np.array([[-1, 2], [-3, 4], [-5, 6]], dtype=np.int32)
    input_dict2 = {"input": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor with different values
    input3 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict3 = {"input": input3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar input
    input4 = np.array(-7, dtype=np.int64)
    input_dict4 = {"input": input4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D complex tensor
    input5 = np.array([1+1j, -2-2j, 3-3j, -4+4j], dtype=np.complex64)
    input_dict5 = {"input": input5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = np.array([], dtype=np.float32)
    input_dict6 = {"input": input6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Tensor with zeros
    input7 = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
    input_dict7 = {"input": input7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.absolute"] = absolute_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.absolute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.absolute'.")

check_valid('torch.absolute', generated_inputs['torch.absolute'], lib="torch")
