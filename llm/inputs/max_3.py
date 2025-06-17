
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_max_3_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, dim=1, keepdim=False
    input_tensor = torch.randn(4, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D int tensor, dim=(0, 2), keepdim=True. This input is problematic, removing it
    # input_tensor = torch.randint(-5, 5, (2, 3, 4)).numpy()
    # input_dict = {
    #     "input": input_tensor,
    #     "dim": (0, 2),
    #     "keepdim": True,
    #     "out": None
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor with only negative values, dim=0, keepdim=True
    input_tensor = torch.rand(3, 3) * -1.0
    input_tensor = input_tensor.numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float tensor, dim=(1, 3), keepdim=False. The dim is invalid, removing it
    # input_tensor = torch.randn(2, 3, 4, 5).numpy()
    # input_dict = {
    #     "input": input_tensor,
    #     "dim": (1, 3),
    #     "keepdim": False,
    #     "out": None
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor, dim=0, keepdim=False
    input_tensor = torch.randn(2, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single tensor
    input_tensor = torch.randn(5).numpy()
    input_dict = {
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.max_3"] = torch_max_3_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.max_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_3'.")

check_valid('torch.max', generated_inputs['torch.max_3'], lib="torch")
