
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logsigmoid_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with negative values
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    input3 = torch.randn(2, 3, 5).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar tensor
    input4 = torch.randn(1).item()
    input_dict4 = {"input": np.array(input4)}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Tensor with large values
    input5 = (torch.randn(2, 2) * 100).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with small values
    input6 = (torch.randn(2, 2) * 0.01).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.logsigmoid"] = logsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.logsigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.logsigmoid'.")

check_valid('torch.nn.functional.logsigmoid', generated_inputs['torch.nn.functional.logsigmoid'], lib="torch")
