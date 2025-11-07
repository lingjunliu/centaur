
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tril_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(3, 3).astype(np.float32)
    diagonal1 = 0
    out1 = np.zeros((3, 3)).astype(np.float32)
    input_dict1 = {"input": input1, "diagonal": diagonal1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(4, 6).astype(np.float64)
    diagonal2 = 1
    out2 = np.zeros((4, 6)).astype(np.float64)
    input_dict2 = {"input": input2, "diagonal": diagonal2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 2).astype(np.float16)
    diagonal3 = -1
    out3 = np.zeros((2, 2)).astype(np.float16)
    input_dict3 = {"input": input3, "diagonal": diagonal3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(5, 5).astype(np.complex128)
    diagonal4 = 2
    out4 = np.zeros((5, 5)).astype(np.complex128)
    input_dict4 = {"input": input4, "diagonal": diagonal4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(1, 1).astype(np.float32)
    diagonal5 = 0
    out5 = np.zeros((1, 1)).astype(np.float32)
    input_dict5 = {"input": input5, "diagonal": diagonal5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 3).astype(np.float64)
    diagonal6 = -2
    out6 = np.zeros((2, 3)).astype(np.float64)
    input_dict6 = {"input": input6, "diagonal": diagonal6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(3, 2).astype(np.float32)
    diagonal7 = 1
    out7 = np.zeros((3, 2)).astype(np.float32)
    input_dict7 = {"input": input7, "diagonal": diagonal7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(4, 4).astype(np.complex64)
    diagonal8 = -1
    out8 = np.zeros((4, 4)).astype(np.complex64)
    input_dict8 = {"input": input8, "diagonal": diagonal8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(5, 5).astype(np.float16)
    diagonal9 = 3
    out9 = np.zeros((5, 5)).astype(np.float16)
    input_dict9 = {"input": input9, "diagonal": diagonal9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 2).astype(np.float64)
    diagonal10 = -2
    out10 = np.zeros((2, 2)).astype(np.float64)
    input_dict10 = {"input": input10, "diagonal": diagonal10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.tril"] = tril_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tril'.")


check_valid('torch.tril', generated_inputs['torch.tril'], lib="torch", suffix=0)
