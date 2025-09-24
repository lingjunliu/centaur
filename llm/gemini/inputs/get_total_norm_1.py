
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def get_total_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    tensors1 = [torch.randn(3, 4).numpy(), torch.randn(5, 2).numpy()]
    input_dict1 = {"parameters": tensors1, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Tensors with negative values
    tensors2 = [torch.randn(2, 2) * -1.0, torch.randn(3, 3) * -1.0]
    tensors2 = [t.numpy() for t in tensors2]
    input_dict2 = {"parameters": tensors2, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different norm_type (inf)
    tensors3 = [torch.randn(4, 4).numpy(), torch.randn(1, 1).numpy()]
    input_dict3 = {"parameters": tensors3, "norm_type": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different norm_type (1.0)
    tensors4 = [torch.randn(4, 4).numpy(), torch.randn(1, 1).numpy()]
    input_dict4 = {"parameters": tensors4, "norm_type": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Empty tensor list
    tensors5 = []
    input_dict5 = {"parameters": tensors5, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 3D tensors
    tensors6 = [torch.randn(2, 3, 4).numpy(), torch.randn(1, 2, 3).numpy()]
    input_dict6 = {"parameters": tensors6, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: norm_type = 0.5
    tensors7 = [torch.randn(2, 3).numpy(), torch.randn(1, 2).numpy()]
    input_dict7 = {"parameters": tensors7, "norm_type": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.utils.get_total_norm_1"] = get_total_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.get_total_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.get_total_norm_1'.")

check_valid('torch.nn.utils.get_total_norm', generated_inputs['torch.nn.utils.get_total_norm_1'], lib="torch")
