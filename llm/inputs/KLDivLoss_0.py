
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np
import torch.nn.functional as F

def KLDivLoss_inputs():
    list_of_inputs = []

    input1 = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1).detach().numpy()
    target1 = F.softmax(torch.rand(3, 5), dim=1).numpy()
    input_dict1 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "log_target": False,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = F.log_softmax(torch.randn(2, 4, requires_grad=True), dim=1).detach().numpy()
    target2 = F.log_softmax(torch.rand(2, 4), dim=1).numpy()
    input_dict2 = {
        "size_average": False,
        "reduce": False,
        "reduction": "batchmean",
        "log_target": True,
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = F.log_softmax(torch.randn(1, 3, 2, requires_grad=True), dim=1).detach().numpy()
    target3 = F.softmax(torch.rand(1, 3, 2), dim=1).numpy()
    input_dict3 = {
        "size_average": None,
        "reduce": None,
        "reduction": "sum",
        "log_target": False,
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = F.log_softmax(torch.randn(4, 6, requires_grad=True), dim=1).detach().numpy()
    target4 = F.softmax(torch.rand(4, 6), dim=1).numpy()
    input_dict4 = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "log_target": False,
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = F.log_softmax(torch.randn(2, 2, requires_grad=True), dim=1).detach().numpy()
    target5 = F.log_softmax(torch.rand(2, 2), dim=1).numpy()
    input_dict5 = {
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "log_target": True,
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.KLDivLoss"] = KLDivLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.KLDivLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.KLDivLoss'.")

check_valid('torch.nn.KLDivLoss', generated_inputs['torch.nn.KLDivLoss'], lib="torch")
