
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_squeeze_inputs():
    list_of_inputs = []

    # Case 1: Basic squeeze with a dimension of size 1
    input1 = torch.randn(2, 1, 3, 1, 4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: No dimension of size 1, so no change
    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Specific dimension to squeeze
    input3 = torch.randn(2, 1, 3, 1, 4).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Multiple dimensions of size 1
    input4 = torch.randn(1, 2, 1, 3, 1).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Integer tensor
    input5 = torch.randint(0, 10, (2, 1, 3)).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Case 6: Complex tensor
    input6 = torch.randn(2, 1, 3, dtype=torch.complex64).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: Zero-dimensional tensor (scalar)
    input7 = torch.tensor(1).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Case 8: Tensor with large values
    input8 = torch.randn(1, 1000, 1).numpy()
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.squeeze_1"] = torch_squeeze_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.squeeze_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.squeeze_1'.")

check_valid('torch.squeeze', generated_inputs['torch.squeeze_1'], lib="torch")
