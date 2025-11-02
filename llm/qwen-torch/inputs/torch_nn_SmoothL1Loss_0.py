
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def smoothl1loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    target = torch.tensor([0.1, 0.2, 0.3]).numpy() # tensor
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    target = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[-1.0, 2.0, -3.0]]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    target = torch.tensor([0.5, 0.5, 0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "beta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "beta": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((1, 2, 3)).numpy()
    target = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]]]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'sum',
        "beta": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0]).numpy()
    target = torch.tensor([0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "beta": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    target = torch.tensor([0.1, 0.2]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": False,
        "reduction": 'sum',
        "beta": 3.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'none',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-2.0, -1.0]).numpy()
    target = torch.tensor([-0.5, 0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smoothl1loss_inputs()

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
