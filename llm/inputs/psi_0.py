
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def psi_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with positive values, avoiding non-positive values
    input2 = np.array([0.5, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional float tensor
    input3 = np.random.rand(2, 3, 4).astype(np.float32) + 1
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Integer tensor, ensuring positive values
    input4 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large float values
    input5 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 2D Integer Tensor
    input6 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.psi"] = psi_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.psi' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.psi'.")

check_valid('torch.special.psi', generated_inputs['torch.special.psi'], lib="torch")
