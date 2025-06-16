
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def PoissonNLLLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with log_input=True
    input_tensor = torch.randn(5, 2).numpy()
    target_tensor = torch.randint(0, 5, (5, 2)).float().numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-08,
        "reduce": None,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: log_input=False, different reduction
    input_tensor = torch.rand(3, 4).numpy()
    target_tensor = torch.randint(0, 3, (3, 4)).float().numpy()
    input_dict = {
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-06,
        "reduce": None,
        "reduction": 'sum',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: reduction='none', different shape
    input_tensor = torch.randn(2, 3, 4).numpy()
    target_tensor = torch.randint(0, 4, (2, 3, 4)).float().numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-08,
        "reduce": None,
        "reduction": 'none',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D input
    input_tensor = torch.randn(10).numpy()
    target_tensor = torch.randint(0, 5, (10,)).float().numpy()
    input_dict = {
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-07,
        "reduce": None,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different eps value
    input_tensor = torch.randn(4, 2).numpy()
    target_tensor = torch.randint(0, 6, (4, 2)).float().numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-04,
        "reduce": None,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.PoissonNLLLoss"] = PoissonNLLLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.PoissonNLLLoss', generated_inputs['torch.nn.PoissonNLLLoss'], lib="torch")
