
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def cartesian_prod_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor
    tensor1 = torch.tensor([1, 2, 3]).numpy()
    tensor2 = torch.tensor([4, 5]).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensors
    tensor1 = torch.ones((2, 3)).numpy()
    tensor2 = torch.zeros((3, 2)).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensors
    tensor1 = torch.ones((2, 3, 4)).numpy()
    tensor2 = torch.zeros((3, 2, 1)).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D and 2D tensors
    tensor1 = torch.tensor([1, 2]).numpy()
    tensor2 = torch.ones((3, 4)).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensors with negative values
    tensor1 = torch.tensor([-1, 0, 1]).numpy()
    tensor2 = torch.tensor([2, 3]).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensors with negative values
    tensor1 = torch.ones((2, 3)).numpy()
    tensor2 = torch.zeros((3, 2)).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D tensors with float values
    tensor1 = torch.tensor([1.5, 2.0]).numpy()
    tensor2 = torch.tensor([3.0, 4.5]).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensors with different shapes
    tensor1 = torch.ones((2, 3, 4)).numpy()
    tensor2 = torch.zeros((3, 2, 1)).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: mixed dimensions
    tensor1 = torch.tensor([1, 2, 3]).numpy()
    tensor2 = torch.ones((3, 4)).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D tensors with different dtypes
    tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    tensor2 = torch.tensor([4, 5], dtype=torch.float32).numpy()
    input_dict = {
        "tensors": [tensor1, tensor2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.cartesian_prod"] = cartesian_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cartesian_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cartesian_prod'.")


check_valid('torch.cartesian_prod', generated_inputs['torch.cartesian_prod'], lib="torch", suffix=0)
