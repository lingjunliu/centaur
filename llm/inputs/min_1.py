
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_min_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor
    input_tensor = torch.randn(5).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor with negative values
    input_tensor = torch.randn(2, 3, 5) * -1.0
    input_tensor = input_tensor.numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Integer tensor
    input_tensor = torch.randint(0, 10, (4, 4)).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Tensor with all same values
    input_tensor = np.full((2,3), 5.0)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Larger tensor
    input_tensor = torch.randn(10, 5, 2).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Tensor with zeros
    input_tensor = torch.zeros(3, 3).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.min_1"] = torch_min_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.min_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_1'.")

check_valid('torch.min', generated_inputs['torch.min_1'], lib="torch")
