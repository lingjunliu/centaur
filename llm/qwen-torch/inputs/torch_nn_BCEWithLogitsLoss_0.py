
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def bcewithlogitsloss_inputs():
    list_of_inputs = []

    
    # Input 1 - Basic case with default parameters
    input = torch.randn((3, 4)).numpy()
    target = torch.randint(0, 2, (3, 4)).numpy().astype('float32')
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - With weight parameter
    input = torch.randn((2, 3)).numpy()
    target = torch.randint(0, 2, (2, 3)).numpy().astype('float32')
    weight = torch.ones((2)).numpy()
    input_dict = {
        "weight": weight,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With pos_weight parameter
    input = torch.randn((2, 3)).numpy()
    target = torch.randint(0, 2, (2, 3)).numpy().astype('float32')
    pos_weight = torch.ones((3)).numpy()
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": pos_weight,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With size_average=False
    input = torch.randn((2, 3)).numpy()
    target = torch.randint(0, 2, (2, 3)).numpy().astype('float32')
    input_dict = {
        "weight": None,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With reduce=False
    input = torch.randn((2, 3)).numpy()
    target = torch.randint(0, 2, (2, 3)).numpy().astype('float32')
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With reduction='none'
    input = torch.randn((2, 3)).numpy()
    target = torch.randint(0, 2, (2, 3)).numpy().astype('float32')
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'none',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Multi-dimensional input with different shapes
    input = torch.randn((1, 2, 3)).numpy()
    target = torch.randint(0, 2, (1, 2, 3)).numpy().astype('float32')
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - With negative values in input tensor
    input = torch.randn((2, 3)).numpy()
    target = torch.randint(0, 2, (2, 3)).numpy().astype('float32')
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - With different reduction types
    input = torch.randn((2, 3)).numpy()
    target = torch.randint(0, 2, (2, 3)).numpy().astype('float32')
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'sum',
        "pos_weight": None,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With pos_weight for multi-class scenario
    input = torch.randn((2, 3, 4)).numpy()
    target = torch.randint(0, 2, (2, 3, 4)).numpy().astype('float32')
    pos_weight = torch.ones((3)).numpy()
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": pos_weight,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.BCEWithLogitsLoss"] = bcewithlogitsloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BCEWithLogitsLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCEWithLogitsLoss'.")


check_valid('torch.nn.BCEWithLogitsLoss', generated_inputs['torch.nn.BCEWithLogitsLoss'], lib="torch", suffix=0)
