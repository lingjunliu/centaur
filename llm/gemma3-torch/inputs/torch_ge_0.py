
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_ge_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3]).numpy()
    other1 = torch.tensor([1, 2, 3]).numpy()
    out1 = torch.tensor([False, False, False]).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other2 = torch.tensor([[1, 1], [4, 4]]).numpy()
    out2 = torch.tensor([[True, True], [False, True]]).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[-1, -2], [3, 4]]).numpy()
    other3 = torch.tensor([0, 0]).numpy()
    out3 = torch.tensor([False, False, True, True]).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out4 = torch.tensor([True, True, True]).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2).numpy()
    other5 = torch.randn(2, 2).numpy()
    out5 = torch.zeros((2, 2)).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([5]).numpy()
    other6 = torch.tensor([2]).numpy()
    out6 = torch.tensor([True]).numpy()
    input_dict6 = {"input": input6, "other": other6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1, 2, 3, 4]).numpy()
    other7 = torch.tensor([4, 3, 2, 1]).numpy()
    out7 = torch.tensor([False, False, True, True]).numpy()
    input_dict7 = {"input": input7, "other": other7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([0, 0, 0]).numpy()
    other8 = torch.tensor([0, 0, 0]).numpy()
    out8 = torch.tensor([True, True, True]).numpy()
    input_dict8 = {"input": input8, "other": other8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([[-1, -2, -3]]).numpy()
    other9 = torch.tensor([[-2, -1, 0]]).numpy()
    out9 = torch.tensor([[True, True, False]]).numpy()
    input_dict9 = {"input": input9, "other": other9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([1, 2, 3]).numpy()
    other10 = torch.tensor([1, 2, 3]).numpy()
    out10 = torch.tensor([True, True, True]).numpy()
    input_dict10 = {"input": input10, "other": other10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.ge"] = torch_ge_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ge' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ge'.")


check_valid('torch.ge', generated_inputs['torch.ge'], lib="torch", suffix=0)
