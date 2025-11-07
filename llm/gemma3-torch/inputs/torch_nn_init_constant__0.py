
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def constant_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3], dtype=np.float32)
    val1 = 0.5
    
    input_dict1 = {
        "tensor": input1,
        "val": val1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    val2 = -1.0
    
    input_dict2 = {
        "tensor": input2,
        "val": val2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.zeros((3, 4, 5), dtype=np.float32)
    val3 = 2.71828
    
    input_dict3 = {
        "tensor": input3,
        "val": val3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.ones((2, 2), dtype=np.float16)
    val4 = -0.1
    
    input_dict4 = {
        "tensor": input4,
        "val": val4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(5, 5).astype(np.float32)
    val5 = 10.0
    
    input_dict5 = {
        "tensor": input5,
        "val": val5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0], dtype=np.float64)
    val6 = 0.0
    
    input_dict6 = {
        "tensor": input6,
        "val": val6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.full((1, 2, 3), 3.14, dtype=np.float32)
    val7 = -2.0
    
    input_dict7 = {
        "tensor": input7,
        "val": val7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([], dtype=np.float32).reshape(0, 0)
    val8 = 1.5
    
    input_dict8 = {
        "tensor": input8,
        "val": val8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.arange(10, dtype=np.float32).reshape(2, 5)
    val9 = -5.0
    
    input_dict9 = {
        "tensor": input9,
        "val": val9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.linspace(0, 1, 10, dtype=np.float32).reshape(5, 2)
    val10 = 7.0
    
    input_dict10 = {
        "tensor": input10,
        "val": val10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.init.constant_"] = constant_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.constant_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.constant_'.")


check_valid('torch.nn.init.constant_', generated_inputs['torch.nn.init.constant_'], lib="torch", suffix=0)
