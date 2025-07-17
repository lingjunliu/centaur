
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_grad_value__inputs():
    list_of_inputs = []

    # Input 1
    p1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    p1.grad = torch.tensor([10.0, -20.0, 30.0])
    parameters = [p1.detach().numpy()]
    clip_value = 15.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    p1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    p1.grad = torch.tensor([[10.0, -20.0], [30.0, -40.0]])
    parameters = [p1.detach().numpy()]
    clip_value = 25.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    p1 = torch.tensor([1.0], requires_grad=True)
    p1.grad = torch.tensor([100.0])
    parameters = [p1.detach().numpy()]
    clip_value = 50.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    p1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    p1.grad = torch.tensor([-10.0, -20.0, -30.0])
    parameters = [p1.detach().numpy()]
    clip_value = 25.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    p1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    p1.grad = torch.tensor([10.0, 20.0, 30.0])
    parameters = [p1.detach().numpy()]
    clip_value = 35.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    p1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    p1.grad = torch.tensor([10.0, -20.0, 30.0])
    p2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    p2.grad = torch.tensor([-40.0, 50.0, -60.0])
    parameters = [p1.detach().numpy(), p2.detach().numpy()]
    clip_value = 45.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    p1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    p1.grad = torch.tensor([10.0, -20.0, 30.0])
    parameters = [p1.detach().numpy()]
    clip_value = 1.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    p1 = torch.randn((2, 3, 4), requires_grad=True)
    p1.grad = torch.randn((2, 3, 4))
    parameters = [p1.detach().numpy()]
    clip_value = 0.5
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    p1 = torch.zeros((2, 2), requires_grad=True)
    p1.grad = torch.ones((2, 2)) * 100
    parameters = [p1.detach().numpy()]
    clip_value = 5.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    p1 = torch.ones((1, 5), requires_grad=True)
    p1.grad = torch.linspace(-100, 100, 5).reshape((1, 5))
    parameters = [p1.detach().numpy()]
    clip_value = 75.0
    input_dict = {"parameters": parameters, "clip_value": clip_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
