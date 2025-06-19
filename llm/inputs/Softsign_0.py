
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def softsign_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive and negative floats
    input1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with mixed values
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with large values
    input3 = (torch.randn(2, 3, 5) * 100).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar tensor
    input4 = torch.tensor(5.0).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Empty tensor
    input5 = torch.empty(0).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with integer values
    input6 = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with only zeros
    input7 = torch.zeros((4, 4)).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Softsign"] = softsign_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softsign'.")

check_valid('torch.nn.Softsign', generated_inputs['torch.nn.Softsign'], lib="torch")
