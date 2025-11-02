
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def cartesian_prod_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor_list = [torch.tensor([0, 1]).numpy(), torch.tensor([2, 3]).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor_list = [torch.tensor([1.0, 2.0]).numpy(), torch.tensor([3.0, 4.0]).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor_list = [torch.tensor([0, 1, 2]).numpy(), torch.tensor([3, 4]).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor_list = [torch.tensor([-1, 0, 1]).numpy(), torch.tensor([2, 3, 4]).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor_list = [torch.ones((2, 3)).numpy(), torch.zeros((2, 3)).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor_list = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy(), torch.tensor([6, 7, 8]).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor_list = [torch.tensor([0.1, 0.2]).numpy(), torch.tensor([0.3, 0.4]).numpy(), torch.tensor([0.5, 0.6]).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor_list = [torch.ones((1, 2)).numpy(), torch.zeros((1, 3)).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor_list = [torch.tensor([0]).numpy(), torch.tensor([1]).numpy(), torch.tensor([2]).numpy(), torch.tensor([3]).numpy()]
    input_dict = {"tensors": tensor_list}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor_list = [torch.tensor([1, 2]).numpy(), torch.tensor([3, 4]).numpy(), torch.tensor([5, 6]).numpy(), torch.tensor([7, 8]).numpy()]
    input_dict = {"tensors": tensor_list}
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
