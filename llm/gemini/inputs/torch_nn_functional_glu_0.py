
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def glu_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, default dim
    input1 = np.array([1.0, 2.0, 3.0, 4.0])
    dim1 = -1
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=0
    input2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    dim2 = 0
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, dim=1
    input3 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    dim3 = 1
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, dim=0
    input4 = np.random.rand(2, 4, 3)
    dim4 = 0
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor, dim=1
    input5 = np.random.rand(2, 4, 3)
    dim5 = 1
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensor, dim=2
    input6 = np.random.rand(2, 3, 4)
    dim6 = 2
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 1D tensor with negative values
    input7 = np.array([-1.0, 2.0, -3.0, 4.0])
    dim7 = -1
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 2D tensor with negative values
    input8 = np.array([[-1.0, 2.0], [-3.0, 4.0], [-5.0, 6.0], [-7.0, 8.0]])
    dim8 = 0
    input_dict8 = {"input": input8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Larger tensor, dim = -1
    input9 = np.random.rand(10, 20)
    dim9 = -1
    input_dict9 = {"input": input9, "dim": dim9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: 4D Tensor
    input10 = np.random.rand(2, 3, 4, 4)
    dim10 = 3
    input_dict10 = {"input": input10, "dim": dim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: 1D tensor with one element
    input11 = np.array([1.0, 2.0])
    dim11 = -1
    input_dict11 = {"input": input11, "dim": dim11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.glu"] = glu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.glu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.glu'.")

check_valid('torch.nn.functional.glu', generated_inputs['torch.nn.functional.glu'], lib="torch", suffix=0)
