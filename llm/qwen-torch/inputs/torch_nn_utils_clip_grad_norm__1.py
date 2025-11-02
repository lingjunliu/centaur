
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def clip_grad_norm__inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor_list = [torch.tensor([1.0, 2.0, 3.0]).numpy()]
    max_norm = 1.0
    norm_type = 2.0
    error_if_nonfinite = True
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor_list = [torch.tensor([0.1, 0.2, 0.3]).numpy(), torch.tensor([10.0, 20.0, 30.0]).numpy()]
    max_norm = 2.0
    norm_type = 1.0
    error_if_nonfinite = False
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor_list = [torch.ones((2, 3)).numpy()]
    max_norm = 0.5
    norm_type = 2.0
    error_if_nonfinite = True
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor_list = [torch.tensor([[1.0, 2.0],
                                [3.0, 4.0]]).numpy()]
    max_norm = 1.5
    norm_type = 1.0
    error_if_nonfinite = False
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor_list = [torch.zeros((1, 2, 3)).numpy()]
    max_norm = 0.1
    norm_type = 2.0
    error_if_nonfinite = True
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor_list = [torch.tensor([0.5, 1.5, 2.5]).numpy()]
    max_norm = 3.0
    norm_type = 1.5
    error_if_nonfinite = False
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor_list = [torch.tensor([1.0]).numpy(), torch.tensor([2.0]).numpy()]
    max_norm = 1.0
    norm_type = 2.0
    error_if_nonfinite = True
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor_list = [torch.tensor([1.0, 2.0, 3.0]).numpy()]
    max_norm = -1.0
    norm_type = 2.0
    error_if_nonfinite = True
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor_list = [torch.tensor([0.1, 0.2, 0.3]).numpy(), torch.tensor([10.0, 20.0, 30.0]).numpy()]
    max_norm = 2.0
    norm_type = -1.0
    error_if_nonfinite = False
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor_list = [torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]]).numpy()]
    max_norm = 1.0
    norm_type = 1.0
    error_if_nonfinite = True
    
    input_dict = {
        "parameters": tensor_list,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.utils.clip_grad_norm__"] = clip_grad_norm__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.clip_grad_norm__1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_norm__1'.")


check_valid('torch.nn.utils.clip_grad_norm_', generated_inputs['torch.nn.utils.clip_grad_norm__1'], lib="torch", suffix=1)
