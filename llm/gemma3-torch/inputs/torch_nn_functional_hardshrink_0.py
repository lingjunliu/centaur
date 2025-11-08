
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hardshrink_inputs():
    list_of_inputs = []
    
    input1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    lambd1 = 0.5
    input_dict1 = {"input": input1, "lambd": lambd1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([0.2, -0.8, 1.5, -2.1], dtype=np.float32)
    lambd2 = 0.7
    input_dict2 = {"input": input2, "lambd": lambd2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    lambd3 = 0.3
    input_dict3 = {"input": input3, "lambd": lambd3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    lambd4 = 0.9
    input_dict4 = {"input": input4, "lambd": lambd4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-0.9, 0.1, -0.5, 0.7], dtype=np.float32)
    lambd5 = 0.0
    input_dict5 = {"input": input5, "lambd": lambd5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([2.5, -1.2, 0.8, -3.1], dtype=np.float32)
    lambd6 = 1.0
    input_dict6 = {"input": input6, "lambd": lambd6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    lambd7 = 0.25
    input_dict7 = {"input": input7, "lambd": lambd7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lambd8 = 0.6
    input_dict8 = {"input": input8, "lambd": lambd8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.1, -2.2, 3.3], [-4.4, 5.5, -6.6]], dtype=np.float32)
    lambd9 = 0.8
    input_dict9 = {"input": input9, "lambd": lambd9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([100.0, -50.0, 25.0], dtype=np.float32)
    lambd10 = 0.4
    input_dict10 = {"input": input10, "lambd": lambd10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardshrink"] = hardshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hardshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardshrink'.")


check_valid('torch.nn.functional.hardshrink', generated_inputs['torch.nn.functional.hardshrink'], lib="torch", suffix=0)
