
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def float_power_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    exponent1 = 2.0
    input_dict1 = {"input": input1, "exponent": exponent1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    exponent2 = 0.5
    input_dict2 = {"input": input2, "exponent": exponent2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([-1.0, -2.0, -3.0])
    exponent3 = 3.0
    input_dict3 = {"input": input3, "exponent": exponent3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0, 1.0, 2.0])
    exponent4 = -1.0
    input_dict4 = {"input": input4, "exponent": exponent4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.5, 2.5, 3.5])
    exponent5 = 1.5
    input_dict5 = {"input": input5, "exponent": exponent5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    exponent6 = 2.0
    input_dict6 = {"input": input6, "exponent": exponent6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0])
    exponent7 = 0.0
    input_dict7 = {"input": input7, "exponent": exponent7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([2.0, 4.0, 8.0])
    exponent8 = -0.5
    input_dict8 = {"input": input8, "exponent": exponent8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.1, 2.2, 3.3])
    exponent9 = 1.0
    input_dict9 = {"input": input9, "exponent": exponent9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.5, 1.5, 2.5])
    exponent10 = 2.5
    input_dict10 = {"input": input10, "exponent": exponent10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.float_power"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.float_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.float_power'.")


check_valid('torch.float_power', generated_inputs['torch.float_power'], lib="torch", suffix=0)
