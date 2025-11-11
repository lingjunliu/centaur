
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def kl_div_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'mean'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'sum'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    target = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]).numpy()
    reduction = 'none'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.1, 0.2]).numpy()
    target = torch.tensor([0.1, 0.2]).numpy()
    reduction = 'mean'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((1, 4)).numpy()
    target = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                          [0.5, 0.6, 0.7, 0.8]]).numpy()
    reduction = 'sum'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    target = torch.tensor([0.1, 0.2]).numpy()
    reduction = 'none'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    reduction = 'mean'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    target = torch.tensor([0.1, 0.2]).numpy()
    reduction = 'sum'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 2)).numpy()
    target = torch.tensor([[0.1, 0.2],
                          [0.3, 0.4],
                          [0.5, 0.6]]).numpy()
    reduction = 'none'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.1]).numpy()
    target = torch.tensor([0.1]).numpy()
    reduction = 'mean'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.kl_div"] = kl_div_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.kl_div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.kl_div'.")


check_valid('torch.nn.functional.kl_div', generated_inputs['torch.nn.functional.kl_div'], lib="torch", suffix=0)
