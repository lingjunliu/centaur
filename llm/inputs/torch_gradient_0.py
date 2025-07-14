
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_gradient_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, default spacing and edge_order
    input1 = torch.tensor([1.0, 4.0, 9.0, 16.0]).numpy()
    spacing1 = [1.0]
    dim1 = [0]
    edge_order1 = 1
    input_dict1 = {"input": input1, "spacing": spacing1, "dim": dim1, "edge_order": edge_order1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, spacing as a scalar
    input2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    spacing2 = [2.0]
    dim2 = [0]
    edge_order2 = 2
    input_dict2 = {"input": input2, "spacing": spacing2, "dim": dim2, "edge_order": edge_order2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, spacing as a list of scalars
    input3 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    spacing3 = [2.0, 3.0]
    dim3 = [0,1]
    edge_order3 = 1
    input_dict3 = {"input": input3, "spacing": spacing3, "dim": dim3, "edge_order": edge_order3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, spacing as a tensor
    input4 = torch.tensor([1.0, 2.0, 4.0, 8.0]).numpy()
    spacing4 = [torch.tensor([0.0, 1.0, 2.0, 3.0]).numpy()]
    dim4 = [0]
    edge_order4 = 2
    input_dict4 = {"input": input4, "spacing": spacing4, "dim": dim4, "edge_order": edge_order4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor, spacing as a list of tensors
    input5 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    spacing5 = [torch.tensor([0.0, 1.0]).numpy(), torch.tensor([0.0, 1.0, 2.0]).numpy()]
    dim5 = [0,1]
    edge_order5 = 1
    input_dict5 = {"input": input5, "spacing": spacing5, "dim": dim5, "edge_order": edge_order5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensor, spacing as a scalar. Using only one dimension to avoid errors.
    input6 = torch.arange(24).reshape(2, 3, 4).numpy()
    spacing6 = [0.5]
    dim6 = [0]
    edge_order6 = 2
    input_dict6 = {"input": input6, "spacing": spacing6, "dim": dim6, "edge_order": edge_order6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 1D tensor with negative spacing
    input7 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    spacing7 = [-1.0]
    dim7 = [0]
    edge_order7 = 1
    input_dict7 = {"input": input7, "spacing": spacing7, "dim": dim7, "edge_order": edge_order7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 2D tensor, different edge_order
    input8 = torch.tensor([[1, 2], [3, 4]]).numpy()
    spacing8 = [1.0,1.0]
    dim8 = [0,1]
    edge_order8 = 2
    input_dict8 = {"input": input8, "spacing": spacing8, "dim": dim8, "edge_order": edge_order8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: 3D tensor, different spacing values. Limiting dimensions to 1 to avoid errors
    input9 = torch.arange(8).reshape(2, 2, 2).numpy()
    spacing9 = [2.0]
    dim9 = [0]
    edge_order9 = 1
    input_dict9 = {"input": input9, "spacing": spacing9, "dim": dim9, "edge_order": edge_order9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 2D tensor, only dim 0
    input10 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    spacing10 = [2.0]
    dim10 = [0]
    edge_order10 = 1
    input_dict10 = {"input": input10, "spacing": spacing10, "dim": dim10, "edge_order": edge_order10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: 2D tensor, specifying both dims
    input11 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    spacing11 = [2.0, 3.0]
    dim11 = [0, 1]
    edge_order11 = 1
    input_dict11 = {"input": input11, "spacing": spacing11, "dim": dim11, "edge_order": edge_order11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.gradient"] = torch_gradient_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gradient'.")

check_valid('torch.gradient', generated_inputs['torch.gradient'], lib="torch", suffix=0)
