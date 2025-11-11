
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gradient_inputs():
    list_of_inputs = []
    
    # Input 1 - 1D tensor with scalar spacing
    input = torch.tensor([4., 1., 1., 16.]).numpy()
    spacing = torch.tensor([-2., -1., 1., 4.]).numpy()
    dim = None
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
    spacing = torch.tensor([2.]).numpy()
    dim = None
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 2D tensor with list of scalars spacing
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = torch.tensor([3., 2.]).numpy()
    dim = None
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 2D tensor with explicit coordinates spacing (both tensors)
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = torch.tensor([0, 2]).numpy()
    dim = None
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - 3D tensor with scalar spacing
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    spacing = torch.tensor([2.]).numpy()
    dim = None
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - 1D tensor with negative values
    input = torch.tensor([-4., -1., 1., 16.]).numpy()
    spacing = torch.tensor([-2., -1., 1., 4.]).numpy()
    dim = None
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - 2D tensor with different dimensions
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = torch.tensor([2., 3.]).numpy()
    dim = 1
    edge_order = 1
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - 2D tensor with different dimensions and edge_order=2
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = torch.tensor([2., 3.]).numpy()
    dim = 0
    edge_order = 2
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - 1D tensor with different spacing values
    input = torch.tensor([1., 4., 9., 16.]).numpy()
    spacing = torch.tensor([1., 2., 3., 4.]).numpy()
    dim = None
    edge_order = 2
    
    input_dict = {
        "input": input,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - 2D tensor with different spacing values and edge_order=2
    input = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = torch.tensor([3., 2.]).numpy()
    dim = None
    edge_order = 2
    
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
