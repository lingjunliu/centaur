
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def round_inputs():
    list_of_inputs = []

    input1 = np.array([4.7, -2.3, 9.1, -7.7])
    decimals1 = 0
    out1 = np.empty_like(input1)

    input_dict1 = {
        "input": input1,
        "decimals": decimals1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-0.5, 0.5, 1.5, 2.5])
    decimals2 = 0
    out2 = np.empty_like(input2)

    input_dict2 = {
        "input": input2,
        "decimals": decimals2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.1234567])
    decimals3 = 3
    out3 = np.empty_like(input3)

    input_dict3 = {
        "input": input3,
        "decimals": decimals3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1200.1234567])
    decimals4 = -3
    out4 = np.empty_like(input4)

    input_dict4 = {
        "input": input4,
        "decimals": decimals4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(2, 2)
    decimals5 = 2
    out5 = np.empty_like(input5)

    input_dict5 = {
        "input": input5,
        "decimals": decimals5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0])
    decimals6 = 0
    out6 = np.empty_like(input6)

    input_dict6 = {
        "input": input6,
        "decimals": decimals6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([-1.0, -2.0, -3.0])
    decimals7 = 1
    out7 = np.empty_like(input7)

    input_dict7 = {
        "input": input7,
        "decimals": decimals7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(3, 4, 5)
    decimals8 = -2
    out8 = np.empty_like(input8)

    input_dict8 = {
        "input": input8,
        "decimals": decimals8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([10000.0], dtype=np.float16)
    decimals9 = 3
    out9 = np.empty_like(input9)

    input_dict9 = {
        "input": input9,
        "decimals": decimals9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([123.456])
    decimals10 = -1
    out10 = np.empty_like(input10)

    input_dict10 = {
        "input": input10,
        "decimals": decimals10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.round"] = round_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.round'.")


check_valid('torch.round', generated_inputs['torch.round'], lib="torch", suffix=0)
