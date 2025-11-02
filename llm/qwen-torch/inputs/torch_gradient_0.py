
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gradient_inputs():
    list_of_inputs = []
    
    # Input 1 - 1D tensor with scalar spacing
    input = torch.tensor([1.0, 4.0, 9.0, 16.0]).numpy()
    spacing = [2.0]
    dim = [0]
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - 2D tensor with scalar spacing
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = [2.0]
    dim = [0, 1]
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 3D tensor with scalar spacing
    input = torch.tensor([[[1, 2], [4, 8]], [[10, 20], [40, 80]]]).numpy()
    spacing = [3.0]
    dim = [0, 1, 2]
    edge_order = 2
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 1D tensor with list spacing
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    spacing = [1.0, 2.0, 3.0]
    dim = [0]
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - 2D tensor with list spacing
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = [3.0, 2.0]
    dim = [0, 1]
    edge_order = 2
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - 1D tensor with negative values
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    spacing = [1.0]
    dim = [0]
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - 2D tensor with negative values
    input = torch.tensor([[-1, -2], [-4, -8]]).numpy()
    spacing = [2.0]
    dim = [0, 1]
    edge_order = 2
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - 3D tensor with negative values
    input = torch.tensor([[[1, -2], [-4, -8]], [[-10, -20], [-40, -80]]]).numpy()
    spacing = [3.0]
    dim = [0, 1, 2]
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - 1D tensor with fractional spacing
    input = torch.tensor([1.0, 4.0, 9.0, 16.0]).numpy()
    spacing = [1.5]
    dim = [0]
    edge_order = 2
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - 2D tensor with fractional spacing
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = [1.5, 2.5]
    dim = [0, 1]
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.gradient"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gradient'.")


check_valid('torch.gradient', generated_inputs['torch.gradient'], lib="torch", suffix=0)
