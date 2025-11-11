
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def smooth_l1_loss_inputs():
    list_of_inputs = []

    
    # Input 1, valid - basic case
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.5, 2.5, 3.5]).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'
    beta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - different beta value
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.5, 2.5, 3.5]).numpy()
    size_average = False
    reduce = False
    reduction = 'none'
    beta = 0.5
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - negative values
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    target = torch.tensor([-1.5, -2.5, -3.5]).numpy()
    size_average = True
    reduce = False
    reduction = 'sum'
    beta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - different dimensions
    input = torch.ones((2, 3)).numpy()
    target = torch.ones((2, 3)).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'
    beta = 0.5
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - large values
    input = torch.tensor([100.0, 200.0, 300.0]).numpy()
    target = torch.tensor([105.0, 205.0, 305.0]).numpy()
    size_average = False
    reduce = True
    reduction = 'none'
    beta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - mixed values
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([-1.5, -2.5, -3.5]).numpy()
    size_average = True
    reduce = False
    reduction = 'sum'
    beta = 2.0
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - different reduction types
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.5, 2.5, 3.5]).numpy()
    size_average = False
    reduce = True
    reduction = 'sum'
    beta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - small beta value
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.5, 2.5, 3.5]).numpy()
    size_average = True
    reduce = False
    reduction = 'none'
    beta = 0.1
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - very large beta value
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.5, 2.5, 3.5]).numpy()
    size_average = False
    reduce = True
    reduction = 'mean'
    beta = 10.0
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - zero beta value
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.5, 2.5, 3.5]).numpy()
    size_average = True
    reduce = False
    reduction = 'sum'
    beta = 0.0
    
    input_dict = {
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "beta": beta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smooth_l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.SmoothL1Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SmoothL1Loss'.")


check_valid('torch.nn.SmoothL1Loss', generated_inputs['torch.nn.SmoothL1Loss'], lib="torch", suffix=0)
