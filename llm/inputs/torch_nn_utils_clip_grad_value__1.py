
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_grad_value__inputs():
    list_of_inputs = []

    # Input 1: Basic test with a single parameter
    p1 = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
    clip_value = 2.0
    input_dict1 = {"parameters": [p1.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multiple parameters
    p21 = torch.tensor([0.5, -1.5, 2.5], requires_grad=True)
    p22 = torch.tensor([[1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    clip_value = 1.0
    input_dict2 = {"parameters": [p21.detach().numpy(), p22.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: No gradients
    p31 = torch.tensor([1.0, 2.0, 3.0])
    clip_value = 0.5
    input_dict3 = {"parameters": [p31.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Large clip value
    p41 = torch.tensor([-5.0, 10.0, -15.0], requires_grad=True)
    clip_value = 20.0
    input_dict4 = {"parameters": [p41.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Negative clip value (should be invalid but testing it)
    p51 = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    clip_value = -1.0
    input_dict5 = {"parameters": [p51.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Zero clip value
    p61 = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    clip_value = 0.0
    input_dict6 = {"parameters": [p61.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Multi-dimensional tensor, different values
    p71 = torch.tensor([[1.5, -2.5], [3.5, -4.5]], requires_grad=True)
    clip_value = 3.0
    input_dict7 = {"parameters": [p71.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Different data types
    p81 = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64, requires_grad=True)
    clip_value = 2.0
    input_dict8 = {"parameters": [p81.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Empty tensor
    p91 = torch.tensor([], requires_grad=True)
    clip_value = 1.0
    input_dict9 = {"parameters": [p91.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Three Parameters with different shapes
    p101 = torch.tensor([1.0, 2.0], requires_grad=True)
    p102 = torch.tensor([[0.5, -1.5], [2.5, -3.5]], requires_grad=True)
    p103 = torch.tensor([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]], requires_grad=True)

    clip_value = 2.5
    input_dict10 = {"parameters": [p101.detach().numpy(), p102.detach().numpy(), p103.detach().numpy()], "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.clip_grad_value__1"] = clip_grad_value__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.clip_grad_value__1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_value__1'.")

check_valid('torch.nn.utils.clip_grad_value_', generated_inputs['torch.nn.utils.clip_grad_value__1'], lib="torch", suffix=1)
