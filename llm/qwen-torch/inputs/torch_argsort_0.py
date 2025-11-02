
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def argsort_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[0.0785, 1.5267, -0.8521, 0.4065],
                        [0.1598, 0.0788, -0.0745, -1.2700],
                        [1.2208, 1.0722, -0.7064, 1.2564],
                        [0.0669, -0.2318, -0.8229, -0.9280]]).numpy()
    dim = 1
    descending = False
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = 0
    descending = True
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[-1.0, 2.0, -3.0, 4.0]]).numpy()
    dim = 0
    descending = False
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 1
    descending = True
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    dim = 0
    descending = False
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = -1
    descending = False
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, -2.0, -3.0, 4.0]]).numpy()
    dim = 0
    descending = True
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]).numpy()
    dim = 0
    descending = False
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    dim = -1
    descending = True
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    dim = 1
    descending = False
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argsort'.")


check_valid('torch.argsort', generated_inputs['torch.argsort'], lib="torch", suffix=0)
