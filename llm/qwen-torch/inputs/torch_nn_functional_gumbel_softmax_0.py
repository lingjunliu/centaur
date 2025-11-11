
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gumbel_softmax_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "tau": 1.0,
        "hard": False,
        "dim": 0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "tau": 0.5,
        "hard": True,
        "dim": 1,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3, 4)).numpy()
    input_dict = {
        "input": input,
        "tau": 1.0,
        "hard": False,
        "dim": 2,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input,
        "tau": 0.1,
        "hard": True,
        "dim": 1,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "tau": 1.0,
        "hard": False,
        "dim": 0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((1, 2, 3, 4)).numpy()
    input_dict = {
        "input": input,
        "tau": 0.5,
        "hard": True,
        "dim": 3,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {
        "input": input,
        "tau": 1.0,
        "hard": False,
        "dim": 0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((3, 2)).numpy()
    input_dict = {
        "input": input,
        "tau": 0.1,
        "hard": True,
        "dim": 1,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).numpy()
    input_dict = {
        "input": input,
        "tau": 0.5,
        "hard": False,
        "dim": 1,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "tau": 0.5,
        "hard": True,
        "dim": 0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.gumbel_softmax"] = gumbel_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.gumbel_softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.gumbel_softmax'.")


check_valid('torch.nn.functional.gumbel_softmax', generated_inputs['torch.nn.functional.gumbel_softmax'], lib="torch", suffix=0)
