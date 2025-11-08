
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def addmv_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mat1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    vec1 = np.array([7.0, 8.0], dtype=np.float32)
    beta1 = 1.0
    alpha1 = 1.0
    out1 = np.array([], dtype=np.float32)
    
    input_dict1 = {
        "input": input1,
        "mat": mat1,
        "vec": vec1,
        "beta": beta1,
        "alpha": alpha1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    mat2 = np.array([[0.5, -0.5], [1.0, 0.0], [-0.5, 0.5]], dtype=np.float64)
    vec2 = np.array([2.0, -1.0], dtype=np.float64)
    beta2 = 0.5
    alpha2 = 2.0
    out2 = np.array([], dtype=np.float64)

    input_dict2 = {
        "input": input2,
        "mat": mat2,
        "vec": vec2,
        "beta": beta2,
        "alpha": alpha2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 1.0], dtype=np.float32)
    mat3 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    vec3 = np.array([0.5, 0.5], dtype=np.float32)
    beta3 = 1.0
    alpha3 = 1.0
    out3 = np.array([], dtype=np.float32)

    input_dict3 = {
        "input": input3,
        "mat": mat3,
        "vec": vec3,
        "beta": beta3,
        "alpha": alpha3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mat4 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    vec4 = np.array([0.5, 1.0], dtype=np.float32)
    beta4 = 1.0
    alpha4 = 1.0
    out4 = np.array([], dtype=np.float32)

    input_dict4 = {
        "input": input4,
        "mat": mat4,
        "vec": vec4,
        "beta": beta4,
        "alpha": alpha4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0], dtype=np.float32)
    mat5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vec5 = np.array([0.5, 1.0], dtype=np.float32)
    beta5 = 1.0
    alpha5 = 1.0
    out5 = np.array([], dtype=np.float32)

    input_dict5 = {
        "input": input5,
        "mat": mat5,
        "vec": vec5,
        "beta": beta5,
        "alpha": alpha5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.addmv"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addmv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv'.")


check_valid('torch.addmv', generated_inputs['torch.addmv'], lib="torch", suffix=0)
