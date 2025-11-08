
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gradient_inputs():
    list_of_inputs = []
    
    # Input 1 - 1D tensor with scalar spacing
    input1d = torch.tensor([1.0, 4.0, 9.0, 16.0]).numpy()
    spacing1d = [2.0]
    dim1d = [0]
    edge_order1d = 1
    
    input_dict1 = {
        "input": input1d,
        "spacing": spacing1d,
        "dim": dim1d,
        "edge_order": edge_order1d
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    # Input 2 - 2D tensor with scalar spacing
    input2d = torch.tensor([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]).numpy()
    spacing2d = [2.0]
    dim2d = [0]
    edge_order2d = 1
    
    input_dict2 = {
        "input": input2d,
        "spacing": spacing2d,
        "dim": dim2d,
        "edge_order": edge_order2d
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3 - 2D tensor with list of scalars spacing
    input2d_2 = torch.tensor([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]).numpy()
    spacing2d_2 = [3.0, 2.0]
    dim2d_2 = [0, 1]
    edge_order2d_2 = 2
    
    input_dict3 = {
        "input": input2d_2,
        "spacing": spacing2d_2,
        "dim": dim2d_2,
        "edge_order": edge_order2d_2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4 - 1D tensor with list of tensors spacing
    input1d_2 = torch.tensor([1.0, 4.0, 9.0, 16.0]).numpy()
    spacing1d_2 = [torch.tensor([0.0, 1.0, 2.0, 3.0])]
    dim1d_2 = [0]
    edge_order1d_2 = 2
    
    input_dict4 = {
        "input": input1d_2,
        "spacing": spacing1d_2,
        "dim": dim1d_2,
        "edge_order": edge_order1d_2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5 - 3D tensor with scalar spacing
    input3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    spacing3d = [2.0]
    dim3d = [0]
    edge_order3d = 1
    
    input_dict5 = {
        "input": input3d,
        "spacing": spacing3d,
        "dim": dim3d,
        "edge_order": edge_order3d
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6 - 1D tensor with negative values and scalar spacing
    input1d_3 = torch.tensor([-1.0, 2.0, -3.0, 4.0]).numpy()
    spacing1d_3 = [1.0]
    dim1d_3 = [0]
    edge_order1d_3 = 1
    
    input_dict6 = {
        "input": input1d_3,
        "spacing": spacing1d_3,
        "dim": dim1d_3,
        "edge_order": edge_order1d_3
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7 - 2D tensor with negative values and scalar spacing
    input2d_3 = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    spacing2d_3 = [1.0]
    dim2d_3 = [0]
    edge_order2d_3 = 1
    
    input_dict7 = {
        "input": input2d_3,
        "spacing": spacing2d_3,
        "dim": dim2d_3,
        "edge_order": edge_order2d_3
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8 - 1D tensor with list of scalars spacing (non-uniform spacing)
    input1d_4 = torch.tensor([1.0, 4.0, 9.0]).numpy()
    spacing1d_4 = [3.0, 2.0]
    dim1d_4 = [0]
    edge_order1d_4 = 2
    
    input_dict8 = {
        "input": input1d_4,
        "spacing": spacing1d_4,
        "dim": dim1d_4,
        "edge_order": edge_order1d_4
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9 - 2D tensor with different dimensions and scalar spacing
    input2d_4 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).numpy()
    spacing2d_4 = [2.0]
    dim2d_4 = [1]
    edge_order2d_4 = 2
    
    input_dict9 = {
        "input": input2d_4,
        "spacing": spacing2d_4,
        "dim": dim2d_4,
        "edge_order": edge_order2d_4
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10 - 3D tensor with scalar spacing and edge_order=2
    input3d_2 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    spacing3d_2 = [2.0]
    dim3d_2 = [0]
    edge_order3d_2 = 2
    
    input_dict10 = {
        "input": input3d_2,
        "spacing": spacing3d_2,
        "dim": dim3d_2,
        "edge_order": edge_order3d_2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
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
