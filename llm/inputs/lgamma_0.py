
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lgamma_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive values
    input1 = torch.tensor([0.5, 1.0, 1.5, 2.0]).numpy()
    out1 = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with positive values
    input2 = torch.tensor([[0.5, 1.0], [1.5, 2.0]]).numpy()
    out2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor with larger positive values
    input3 = torch.tensor([5.0, 10.0, 15.0, 20.0]).numpy()
    out3 = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor with positive values
    input4 = torch.tensor([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]).numpy()
    out4 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor with values close to zero
    input5 = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    out5 = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.tensor([]).numpy()
    out6 = torch.tensor([]).numpy()
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with a mix of small and large numbers
    input7 = torch.tensor([0.01, 1, 10, 100]).numpy()
    out7 = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.lgamma"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lgamma'.")

check_valid('torch.lgamma', generated_inputs['torch.lgamma'], lib="torch", suffix=0)
