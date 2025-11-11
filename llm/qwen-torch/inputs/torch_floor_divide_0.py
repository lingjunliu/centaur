
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def floor_divide_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = torch.tensor([3.0, 4.0, 5.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([100.0]).numpy()
    other = torch.tensor([7.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((3, 4)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-10.0, -20.0, -30.0]).numpy()
    other = torch.tensor([3.0, 4.0, 5.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = torch.tensor([-3.0, -4.0, -5.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 2)).numpy()
    other = torch.tensor([[1.0, 2.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0]).numpy()
    other = torch.tensor([1.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((4, 5)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0],
                        [11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([5.0]).numpy()
    other = torch.tensor([2.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide'], lib="torch", suffix=0)
