
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gradient_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 4.0, 9.0, 16.0])
    spacing_list = [2.0]
    dim_list = [0]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2, 4, 8], [10, 20, 40, 80]])
    spacing_list = [1.0, 2.0]
    dim_list = [0, 1]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1, -2, -3], [-3, -4, -5], [-6, -7, -8]])
    spacing_list = [0.5, 1.5]
    dim_list = [0, 1]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1, 2, 3, 4, 5])
    spacing_list = [0.1]
    dim_list = [0]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    spacing_list = [2, 2]
    dim_list = [0,1]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 2, 3], [3, 4, 5], [5, 6, 7]], [[5, 6, 7], [7, 8, 9], [9, 10, 11]]])
    spacing_list = [1, 2, 0.5]
    dim_list = [0, 1, 2]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1, 2, 3])
    spacing_list = [0.5]
    dim_list = [0]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    spacing_list = [2.0, 1.5]
    dim_list = [0, 1]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.array([1, 2, 3, 4])
    spacing_list = [1.5]
    dim_list = [0]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    spacing_list = [2, 3]
    dim_list = [0, 1]
    edge_order_int = 1
    input_dict = {"input": input_tensor, "spacing": spacing_list, "dim": dim_list, "edge_order": edge_order_int}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.gradient"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gradient'.")

check_valid('torch.gradient', generated_inputs['torch.gradient'], lib="torch", suffix=0)
