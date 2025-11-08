
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def reshape_inputs():
    list_of_inputs = []

    input1 = torch.arange(12.).numpy()
    shape1 = (3, 4)
    input_dict1 = {"input": input1, "shape": shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[0, 1], [2, 3]]).numpy()
    shape2 = (-1,)
    input_dict2 = {"input": input2, "shape": shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 4).numpy()
    shape3 = (6, 4)
    input_dict3 = {"input": input3, "shape": shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1, 2, 3, 4, 5, 6]).numpy()
    shape4 = (2, -1)
    input_dict4 = {"input": input4, "shape": shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randint(0, 10, (5,)).numpy()
    shape5 = (1, 5)
    input_dict5 = {"input": input5, "shape": shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.zeros(2, 2, 2).numpy()
    shape6 = (4,2)
    input_dict6 = {"input": input6, "shape": shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.ones((1, 3, 2)).numpy()
    shape7 = (3, 2)
    input_dict7 = {"input": input7, "shape": shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.rand(4, 4).numpy()
    shape8 = (16,)
    input_dict8 = {"input": input8, "shape": shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    shape9 = (2, 2)
    input_dict9 = {"input": input9, "shape": shape9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reshape'.")


check_valid('torch.reshape', generated_inputs['torch.reshape'], lib="torch", suffix=0)
