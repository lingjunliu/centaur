
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_dot_inputs():
    list_of_inputs = []

    input1 = torch.tensor([2, 3]).numpy()
    input2 = torch.tensor([2, 1]).numpy()
    input_dict = {"input": input1, "tensor": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([0, 1]).numpy()
    input2 = torch.tensor([2, 3]).numpy()
    input_dict = {"input": input1, "tensor": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([-1, 2, -3]).numpy()
    input2 = torch.tensor([4, -5, 6]).numpy()
    input_dict = {"input": input1, "tensor": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input2 = torch.tensor([4.5, 5.5, 6.5]).numpy()
    input_dict = {"input": input1, "tensor": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    input2 = torch.tensor([4, 5, 6], dtype=torch.int64).numpy()
    input_dict = {"input": input1, "tensor": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input2 = torch.tensor([0.4, 0.5, 0.6]).numpy()
    out_tensor = torch.empty(1).numpy()
    input_dict = {"input": input1, "tensor": input2, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([-2.0, 3.0, -1.0]).numpy()
    input2 = torch.tensor([1.0, -2.0, 3.0]).numpy()
    input_dict = {"input": input1, "tensor": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dot"] = torch_dot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dot'.")

check_valid('torch.dot', generated_inputs['torch.dot'], lib="torch")
