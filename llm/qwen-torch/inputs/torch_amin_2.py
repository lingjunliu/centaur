
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def amin_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[0.6451, -0.4866, 0.2987, -1.3312],
                        [-0.5744, 1.2980, 1.8397, -0.2713],
                        [0.9128, 0.9214, -1.7268, -0.2995],
                        [0.9023, 0.4853, 0.9075, -1.6165]]).numpy()
    dim = (1,) # tuple
    keepdim = False # boolean
    out = torch.empty(4).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[-0.5744, 1.2980, 1.8397, -0.2713],
                        [0.9128, 0.9214, -1.7268, -0.2995],
                        [0.9023, 0.4853, 0.9075, -1.6165]]).numpy()
    dim = (0,) # tuple
    keepdim = True # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[-0.5744, 1.2980, 1.8397, -0.2713],
                        [0.9128, 0.9214, -1.7268, -0.2995],
                        [0.9023, 0.4853, 0.9075, -1.6165]]).numpy()
    dim = (0, 1) # tuple
    keepdim = False # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[-0.5744, 1.2980, 1.8397, -0.2713],
                        [0.9128, 0.9214, -1.7268, -0.2995],
                        [0.9023, 0.4853, 0.9075, -1.6165]]).numpy()
    dim = (0, 1) # tuple
    keepdim = True # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0.6451, -0.4866, 0.2987],
                        [-0.5744, 1.2980, 1.8397],
                        [0.9128, 0.9214, -1.7268]]).numpy()
    dim = (2,) # tuple
    keepdim = False # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.6451, -0.4866, 0.2987],
                        [-0.5744, 1.2980, 1.8397],
                        [0.9128, 0.9214, -1.7268]]).numpy()
    dim = (2,) # tuple
    keepdim = True # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-0.5744, 1.2980, 1.8397],
                        [0.9128, 0.9214, -1.7268],
                        [0.9023, 0.4853, 0.9075]]).numpy()
    dim = (0, 1) # tuple
    keepdim = False # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[-0.5744, 1.2980, 1.8397],
                        [0.9128, 0.9214, -1.7268],
                        [0.9023, 0.4853, 0.9075]]).numpy()
    dim = (0, 1) # tuple
    keepdim = True # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[-0.5744, 1.2980, 1.8397],
                        [0.9128, 0.9214, -1.7268],
                        [0.9023, 0.4853, 0.9075]]).numpy()
    dim = (0,) # tuple
    keepdim = False # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[-0.5744, 1.2980, 1.8397],
                        [0.9128, 0.9214, -1.7268],
                        [0.9023, 0.4853, 0.9075]]).numpy()
    dim = (1,) # tuple
    keepdim = False # boolean
    out = torch.empty(3).numpy() # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.amin_2"] = amin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.amin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin_2'.")


check_valid('torch.amin', generated_inputs['torch.amin_2'], lib="torch", suffix=2)
