
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_power_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n1 = 2
    input_dict1 = {"input": input1, "n": n1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    n2 = 5
    input_dict2 = {"input": input2, "n": n2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1.0, 0.0], [0.0, -1.0]])
    n3 = 3
    input_dict3 = {"input": input3, "n": n3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n4 = 0
    input_dict4 = {"input": input4, "n": n4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[0.0, 1.0], [1.0, 0.0]])
    n5 = 4
    input_dict5 = {"input": input5, "n": n5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[2.0, 1.0], [1.0, 2.0]])
    n6 = 2
    input_dict6 = {"input": input6, "n": n6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[0.5, 0.2], [0.1, 0.8]])
    n7 = 3
    input_dict7 = {"input": input7, "n": n7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1.1, 2.2], [3.3, 4.4]])
    n8 = 2
    input_dict8 = {"input": input8, "n": n8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n9 = -1
    input_dict9 = {"input": input9, "n": n9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.matrix_power"] = matrix_power_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.matrix_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.matrix_power'.")


check_valid('torch.matrix_power', generated_inputs['torch.matrix_power'], lib="torch", suffix=0)
