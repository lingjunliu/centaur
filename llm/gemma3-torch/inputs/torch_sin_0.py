
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_sin_inputs():
    list_of_inputs = []

    input1 = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    out1 = np.zeros_like(input1)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-np.pi/2, -np.pi, -3*np.pi/2])
    out2 = np.zeros_like(input2)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]])
    out3 = np.zeros_like(input3)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-np.pi/2, -np.pi], [-3*np.pi/2, -2*np.pi]])
    out4 = np.zeros_like(input4)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0])
    out5 = np.zeros_like(input5)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([-1.0, -2.0, -3.0])
    out6 = np.zeros_like(input6)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[0.0, np.pi/2], [np.pi, 3*np.pi/2]], [[-np.pi/2, -np.pi], [-3*np.pi/2, -2*np.pi]]])
    out7 = np.zeros_like(input7)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(5, 5)
    out8 = np.zeros_like(input8)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.randn(2, 3, 4)
    out9 = np.zeros_like(input9)
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.0])
    out10 = np.zeros_like(input10)
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.sin"] = torch_sin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sin'.")


check_valid('torch.sin', generated_inputs['torch.sin'], lib="torch", suffix=0)
