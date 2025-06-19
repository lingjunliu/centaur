
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bceloss_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([0.2, 0.8, 0.4]).numpy()
    target = torch.tensor([0.0, 1.0, 0.5]).numpy()
    weight = torch.tensor([1.0, 1.0, 1.0]).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'
    input_dict = {
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(2, 3, requires_grad=True).sigmoid().detach().numpy()
    target = torch.rand(2, 3, requires_grad=False).numpy()
    weight = torch.ones(2,3).numpy()
    size_average = False
    reduce = False
    reduction = 'sum'
    input_dict = {
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(1, 5).sigmoid().detach().numpy()
    target = torch.rand(1, 5).numpy()
    weight = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]).numpy()
    size_average = None
    reduce = None
    reduction = 'none'
    input_dict = {
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = torch.tensor([[0.1, 0.9], [0.3, 0.7]]).numpy()
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).numpy()
    weight = torch.tensor([1.0, 2.0]).numpy()
    size_average = True
    reduce = False
    reduction = 'mean'
    input_dict = {
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.rand(4, 2, 2).sigmoid().detach().numpy()
    target = torch.randint(0, 2, (4, 2, 2)).float().numpy()
    weight = torch.ones(4, 2, 2).numpy()
    size_average = False
    reduce = True
    reduction = 'sum'
    input_dict = {
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.BCELoss"] = bceloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.BCELoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCELoss'.")

check_valid('torch.nn.BCELoss', generated_inputs['torch.nn.BCELoss'], lib="torch", suffix=0)
