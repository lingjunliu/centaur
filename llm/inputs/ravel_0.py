
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ravel_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor
    input_1 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Example 2: 2D tensor
    input_2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Example 3: 3D tensor with different values
    input_3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Example 4: Tensor with negative values
    input_4 = torch.tensor([[-1, 2], [-3, 4]]).numpy()
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Example 5: Tensor with float values
    input_5 = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Example 6: Tensor with complex values
    input_6 = torch.tensor([[1+1j, 2+2j], [3+3j, 4+4j]]).numpy()
    input_dict_6 = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Example 7: Empty tensor
    input_7 = torch.tensor([]).numpy()
    input_dict_7 = {"input": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Example 8: Higher dimensional tensor
    input_8 = torch.randn(2, 3, 4, 5).numpy()
    input_dict_8 = {"input": input_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs["torch.ravel"] = ravel_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ravel'.")

check_valid('torch.ravel', generated_inputs['torch.ravel'], lib="torch")
