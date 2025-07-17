
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def householder_product_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1.0, 2.0], [4.0, 5.0]])
    h_tensor = np.array([0.1, 0.2])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 0.0], [0.0, 1.0]])
    h_tensor = np.array([1.0, 0.0])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    h_tensor = np.array([-1.0, 1.0])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]], [[7.0, 8.0, 9.0], [9.0, 10.0, 11.0], [11.0, 12.0, 13.0]]])
    h_tensor = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = np.array([[1.0, 2.0]])
    h_tensor = np.array([0.1, 0.1])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0])
    h_tensor = np.array([0.2, 0.4, 0.6, 0.8])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    h_tensor = np.array([0.1, 0.1])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    h_tensor = np.array([0.0, 0.0])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1.0, 2.0, 3.0])
    h_tensor = np.array([1.0, 1.0, 1.0])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    h_tensor = np.array([-1.0, -1.0])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Ensure shape[-2] >= shape[-1] for 3D tensor, and correct h shape
    input_tensor = np.random.rand(2, 3, 2)
    h_tensor = np.random.rand(2, 2)
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Ensure shape[-2] >= shape[-1] for 4D tensor and correct h shape
    input_tensor = np.random.rand(2, 2, 3, 3)
    h_tensor = np.random.rand(2, 2, 3)
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Test with complex numbers
    input_tensor = np.array([[1.0 + 1j, 2.0 - 2j], [3.0 + 0j, 4.0 - 1j]])
    h_tensor = np.array([0.1 + 0j, 0.2 + 0j])
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: Different shape where last two dims are equal
    input_tensor = np.random.rand(2, 2, 2)
    h_tensor = np.random.rand(2, 2)
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15: 5D tensor, making sure the last two dimensions comply with the rule and the h tensor is correct
    input_tensor = np.random.rand(1, 2, 2, 3, 3)
    h_tensor = np.random.rand(1, 2, 2, 3)
    input_dict = {"input": input_tensor, "h": h_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.householder_product"] = householder_product_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.householder_product' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.householder_product'.")

check_valid('torch.linalg.householder_product', generated_inputs['torch.linalg.householder_product'], lib="torch", suffix=0)
