
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def median_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D tensor
    input_tensor = torch.tensor([1.5219, -1.5212, 0.2202]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    input_tensor = torch.randn(4, 5).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Single element tensor
    input_tensor = torch.tensor([5.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Tensor with negative values
    input_tensor = torch.tensor([-5.0, -2.0, -10.0, -1.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Tensor with mixed positive and negative values
    input_tensor = torch.tensor([[0.2505, -0.3982, -0.9948, 0.3518, -1.3131],
                                  [0.3180, -0.6993, 1.0436, 0.0438, 0.2270]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Tensor with even number of elements
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Tensor with odd number of elements
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Tensor with zeros
    input_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Large 1D tensor
    input_tensor = torch.randn(100).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Small 2D tensor
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.median_1"] = median_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.median_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.median_1'.")


check_valid('torch.median', generated_inputs['torch.median_1'], lib="torch", suffix=1)
