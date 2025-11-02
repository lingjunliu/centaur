
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def empty_like_inputs():
    list_of_inputs = []
    
    # Input 1
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input = torch.ones((2, 3)).numpy()
    dtype = torch.float64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input = torch.zeros((1, 2, 3)).numpy()
    dtype = torch.int32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    dtype = torch.int64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = torch.tensor([[-1.0, -2.0, -3.0]]).numpy()
    dtype = torch.float64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = torch.tensor([0.0]).numpy()
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = torch.tensor([1.0, 2.0]).numpy()
    dtype = torch.int32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dtype = torch.float64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dtype = torch.int64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.empty_like"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.empty_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_like'.")


check_valid('torch.empty_like', generated_inputs['torch.empty_like'], lib="torch", suffix=0)
