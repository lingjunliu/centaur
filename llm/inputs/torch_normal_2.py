
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_normal_inputs():
    list_of_inputs = []

    # Input 1
    mean = 0.0
    std = torch.tensor([1.0, 2.0, 3.0])
    out = torch.zeros(3, dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    mean = 1.0
    std = torch.tensor([0.5, 1.5, 2.5, 3.5])
    out = torch.zeros(4, dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    mean = -1.0
    std = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    out = torch.zeros(5, dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    mean = 2.5
    std = torch.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    out = torch.zeros((2, 2), dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    mean = -2.5
    std = torch.tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
    out = torch.zeros((2, 2, 2), dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    mean = 0.7
    std = torch.tensor(np.arange(1.0, 7.0))
    out = torch.zeros(6, dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    mean = -0.7
    std = torch.tensor(np.arange(0.1, 1.1, 0.1))
    out = torch.zeros(10, dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    mean = 10.0
    std = torch.tensor(np.array([1.0]))
    out = torch.zeros(1, dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    mean = -10.0
    std = torch.tensor(np.array([10.0]))
    out = torch.zeros(1, dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    mean = 5.0
    std = torch.tensor(np.random.rand(2, 3))
    out = torch.zeros((2,3), dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    mean = -5.0
    std = torch.tensor(np.random.rand(1, 5))
    out = torch.zeros((1,5), dtype=torch.float32)
    input_dict = {"mean": mean, "std": std.numpy(), "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.normal_2"] = torch_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.normal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_2'.")

check_valid('torch.normal', generated_inputs['torch.normal_2'], lib="torch", suffix=2)
