
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def parameters_to_vector_inputs():
    list_of_inputs = []

    # Test case 1: List of float tensors with different sizes
    params1 = [torch.randn(2, 3).float().numpy(), torch.randn(5).float().numpy(), torch.randn(1, 1, 4).float().numpy()]
    input_dict1 = {"parameters": params1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: List of int tensors with different sizes
    params2 = [torch.randint(0, 10, (2, 2)).int().numpy(), torch.randint(0, 5, (3,)).int().numpy()]
    input_dict2 = {"parameters": params2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: List of mixed type tensors (float, int)
    params3 = [torch.randn(3, 3).float().numpy(), torch.randint(0, 5, (2,)).int().numpy(), torch.randn(1).float().numpy()]
    input_dict3 = {"parameters": params3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: List of tensors with negative values
    params4 = [torch.randn(2, 3).numpy() - 2, torch.randn(5).numpy() - 1]
    input_dict4 = {"parameters": params4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: List of tensors with zero size (removing this as it may cause issues)
    #params5 = [torch.randn(0, 3).float().numpy(), torch.randn(5).float().numpy()]
    #input_dict5 = {"parameters": params5}
    #list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: List containing a single tensor
    params6 = [torch.randn(4, 4).float().numpy()]
    input_dict6 = {"parameters": params6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Test case 7: Empty list (removing this, as it will error)
    #params7 = []
    #input_dict7 = {"parameters": params7}
    #list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.parameters_to_vector"] = parameters_to_vector_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.parameters_to_vector' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.parameters_to_vector'.")

check_valid('torch.nn.utils.parameters_to_vector', generated_inputs['torch.nn.utils.parameters_to_vector'], lib="torch")
