
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pdist_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p1 = 2.0
    
    input_dict1 = {"input": input1, "p": p1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    p2 = 1.0
    
    input_dict2 = {"input": input2, "p": p2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    p3 = 2.0
    
    input_dict3 = {"input": input3, "p": p3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    p4 = 3.0
    
    input_dict4 = {"input": input4, "p": p4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0], [2.0], [3.0]])
    p5 = 2.0
    
    input_dict5 = {"input": input5, "p": p5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    p6 = 0.5
    
    input_dict6 = {"input": input6, "p": p6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    p7 = 3.0
    
    input_dict7 = {"input": input7, "p": p7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p8 = np.inf
    
    input_dict8 = {"input": input8, "p": p8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.5, 2.5], [3.5, 4.5]])
    p9 = 1.5
    
    input_dict9 = {"input": input9, "p": p9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[10.0, 20.0], [30.0, 40.0]])
    p10 = 4.0
    
    input_dict10 = {"input": input10, "p": p10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.pdist"] = pdist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pdist'.")


check_valid('torch.nn.functional.pdist', generated_inputs['torch.nn.functional.pdist'], lib="torch", suffix=0)
