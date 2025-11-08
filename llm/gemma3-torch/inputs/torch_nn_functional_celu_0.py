
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha1 = 1.0
    inplace1 = False
    input_dict1 = {
        "input": input1,
        "alpha": alpha1,
        "inplace": inplace1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    alpha2 = 1.0
    inplace2 = False
    input_dict2 = {
        "input": input2,
        "alpha": alpha2,
        "inplace": inplace2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    alpha3 = 1.0
    inplace3 = False
    input_dict3 = {
        "input": input3,
        "alpha": alpha3,
        "inplace": inplace3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    alpha4 = 1.0
    inplace4 = False
    input_dict4 = {
        "input": input4,
        "alpha": alpha4,
        "inplace": inplace4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha5 = 2.0
    inplace5 = False
    input_dict5 = {
        "input": input5,
        "alpha": alpha5,
        "inplace": inplace5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha6 = 1.0
    inplace6 = True
    input_dict6 = {
        "input": input6,
        "alpha": alpha6,
        "inplace": inplace6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    alpha7 = 2.0
    inplace7 = True
    input_dict7 = {
        "input": input7,
        "alpha": alpha7,
        "inplace": inplace7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    alpha8 = 1.0
    inplace8 = False
    input_dict8 = {
        "input": input8,
        "alpha": alpha8,
        "inplace": inplace8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha9 = 0.5
    inplace9 = False
    input_dict9 = {
        "input": input9,
        "alpha": alpha9,
        "inplace": inplace9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    alpha10 = 1.5
    inplace10 = True
    input_dict10 = {
        "input": input10,
        "alpha": alpha10,
        "inplace": inplace10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.celu'.")


check_valid('torch.nn.functional.celu', generated_inputs['torch.nn.functional.celu'], lib="torch", suffix=0)
