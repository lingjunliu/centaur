
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def tanhshrink_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor
    input1 = np.array(0.5)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with positive values
    input2 = np.array([1.0, 2.0, 3.0])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor with negative values
    input3 = np.array([-1.0, -2.0, -3.0])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor with mixed values
    input4 = np.array([-1.0, 0.0, 1.0])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D tensor with negative values
    input6 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 3D tensor
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Tensor with large values
    input8 = np.array([10.0, 20.0, 30.0])
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Tensor with small values
    input9 = np.array([0.1, 0.2, 0.3])
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Tensor with zeros
    input10 = np.array([0.0, 0.0, 0.0])
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Tensor with a mix of large, small, and zero values
    input11 = np.array([-10.0, 0.01, 0.0, 100.0, -0.01])
    input_dict11 = {"input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.tanhshrink"] = tanhshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.tanhshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.tanhshrink'.")

check_valid('torch.nn.functional.tanhshrink', generated_inputs['torch.nn.functional.tanhshrink'], lib="torch", suffix=0)
