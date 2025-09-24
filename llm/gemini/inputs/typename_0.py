
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def typename_inputs():
    list_of_inputs = []

    # Input 1: Float Tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"obj": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int Tensor
    input2 = torch.randint(0, 10, (2, 2)).numpy()
    input_dict2 = {"obj": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Double Tensor
    input3 = torch.randn(5, 5, dtype=torch.float64).numpy()
    input_dict3 = {"obj": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Long Tensor
    input4 = torch.randint(-10, 10, (1, 5), dtype=torch.int64).numpy()
    input_dict4 = {"obj": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Bool Tensor
    input5 = torch.randint(0, 2, (4, 3), dtype=torch.bool).numpy()
    input_dict5 = {"obj": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Complex Tensor
    input6 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict6 = {"obj": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 3D Float Tensor
    input7 = torch.randn(2, 3, 4).numpy()
    input_dict7 = {"obj": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Empty Tensor
    input8 = torch.empty(0).numpy()
    input_dict8 = {"obj": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Tensor with negative values
    input9 = torch.randn(2, 2) * -10
    input9 = input9.numpy()
    input_dict9 = {"obj": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.typename"] = typename_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.typename' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.typename'.")

check_valid('torch.typename', generated_inputs['torch.typename'], lib="torch")
