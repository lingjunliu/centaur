
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([10, 20, 30]).numpy()
    other1 = np.int32(2)
    
    input_dict1 = {
        "input": input1,
        "other": other1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-10, -20, 30]).numpy()
    other2 = np.int32(3)
    
    input_dict2 = {
        "input": input2,
        "other": other2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([10.5, 20.5, 30.5]).numpy()
    other3 = np.int32(2)
    
    input_dict3 = {
        "input": input3,
        "other": other3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[10, 20], [30, 40]]).numpy()
    other4 = np.int32(5)
    
    input_dict4 = {
        "input": input4,
        "other": other4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    other5 = np.int32(1)
    
    input_dict5 = {
        "input": input5,
        "other": other5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([10, 20, 30]).numpy()
    other6 = np.int64(4)

    input_dict6 = {
        "input": input6,
        "other": other6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other7 = np.int32(2)

    input_dict7 = {
        "input": input7,
        "other": other7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([100, 200, 300]).numpy()
    other8 = np.int32(10)
    
    input_dict8 = {
        "input": input8,
        "other": other8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.floor_divide_1"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_1'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_1'], lib="torch", suffix=1)
