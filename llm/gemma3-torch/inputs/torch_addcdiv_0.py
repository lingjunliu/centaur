
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def addcdiv_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).numpy()
    tensor1_1 = torch.randn(2, 3).numpy()
    tensor2_1 = torch.randn(2, 3).numpy()
    value1 = 0.5
    out1 = torch.empty(2, 3).numpy()

    input_dict1 = {
        "input": input1,
        "tensor1": tensor1_1,
        "tensor2": tensor2_1,
        "value": value1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1_2 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2_2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    value2 = 2.0
    out2 = torch.empty(3).numpy()

    input_dict2 = {
        "input": input2,
        "tensor1": tensor1_2,
        "tensor2": tensor2_2,
        "value": value2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 2).numpy()
    tensor1_3 = torch.randn(3, 2).numpy()
    tensor2_3 = torch.randn(3, 2).numpy()
    value3 = -0.1
    out3 = torch.empty(3, 2).numpy()

    input_dict3 = {
        "input": input3,
        "tensor1": tensor1_3,
        "tensor2": tensor2_3,
        "value": value3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.ones((1, 4)).numpy()
    tensor1_4 = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).numpy()
    tensor2_4 = torch.tensor([[2.0, 2.0, 2.0, 2.0]]).numpy()
    value4 = 1.0
    out4 = torch.empty((1, 4)).numpy()

    input_dict4 = {
        "input": input4,
        "tensor1": tensor1_4,
        "tensor2": tensor2_4,
        "value": value4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.addcdiv"] = addcdiv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addcdiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addcdiv'.")


check_valid('torch.addcdiv', generated_inputs['torch.addcdiv'], lib="torch", suffix=0)
