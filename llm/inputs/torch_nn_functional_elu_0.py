
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def elu_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive values, alpha=1.0, inplace=False
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {"input": input1, "alpha": 1.0, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with negative values, alpha=1.0, inplace=False
    input2 = np.array([-1.0, -2.0, -3.0])
    input_dict2 = {"input": input2, "alpha": 1.0, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor with mixed values, alpha=0.5, inplace=False
    input3 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]])
    input_dict3 = {"input": input3, "alpha": 0.5, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, alpha=2.0, inplace=False
    input4 = np.random.rand(2, 3, 4)
    input_dict4 = {"input": input4, "alpha": 2.0, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 1D tensor with positive values, alpha=1.0, inplace=True
    input5 = np.array([1.0, 2.0, 3.0])
    input_dict5 = {"input": input5, "alpha": 1.0, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D tensor with negative values, alpha=1.0, inplace=True
    input6 = np.array([-1.0, -2.0, -3.0])
    input_dict6 = {"input": input6, "alpha": 1.0, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 2D tensor with mixed values, alpha=0.5, inplace=True
    input7 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]])
    input_dict7 = {"input": input7, "alpha": 0.5, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D tensor, alpha=2.0, inplace=True
    input8 = np.random.rand(2, 3, 4)
    input_dict8 = {"input": input8, "alpha": 2.0, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Single value tensor, alpha=1.0, inplace=False
    input9 = np.array(5.0)
    input_dict9 = {"input": input9, "alpha": 1.0, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Single value tensor, alpha=0.2, inplace=True
    input10 = np.array(-3.0)
    input_dict10 = {"input": input10, "alpha": 0.2, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Empty array, alpha=1.0, inplace=False
    input11 = np.array([])
    input_dict11 = {"input": input11, "alpha": 1.0, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: Array with very large values, alpha=1.0, inplace=False
    input12 = np.array([1e10, -1e10])
    input_dict12 = {"input": input12, "alpha": 1.0, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.elu"] = elu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.elu'.")

check_valid('torch.nn.functional.elu', generated_inputs['torch.nn.functional.elu'], lib="torch", suffix=0)
