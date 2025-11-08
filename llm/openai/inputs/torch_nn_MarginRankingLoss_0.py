
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.tensor([0.2, 1.5, -0.3]).numpy()
    input2 = torch.tensor([0.1, 2.0, -1.0]).numpy()
    target = torch.tensor([1.0, -1.0, 1.0]).numpy()
    input_dict = {
        "margin": 0.0,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).numpy()
    input2 = torch.tensor([-1.5, -1.5, 0.5, 0.5, 1.5]).numpy()
    target = torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0]).numpy()
    input_dict = {
        "margin": 1.0,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input1 = torch.tensor([5.0]).numpy()
    input2 = torch.tensor([5.0]).numpy()
    target = torch.tensor([1.0]).numpy()
    input_dict = {
        "margin": 0.5,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (scalar)
    input1 = torch.tensor(0.0).numpy()
    input2 = torch.tensor(-1.0).numpy()
    target = torch.tensor(1.0).numpy()
    input_dict = {
        "margin": 0.0,
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (float64)
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64).numpy()
    input2 = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float64).numpy()
    target = torch.tensor([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=torch.float64).numpy()
    input_dict = {
        "margin": 0.2,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input1 = torch.tensor([-3.0, 4.0]).numpy()
    input2 = torch.tensor([-4.0, 5.0]).numpy()
    target = torch.tensor([1.0, -1.0]).numpy()
    input_dict = {
        "margin": 2.0,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (zeros)
    input1 = torch.zeros(4).numpy()
    input2 = torch.zeros(4).numpy()
    target = torch.tensor([1.0, -1.0, 1.0, -1.0]).numpy()
    input_dict = {
        "margin": -0.1,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input1 = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.0]).numpy()
    input2 = torch.tensor([0.0, 0.0, 1.0, -1.0, 2.0, -3.0, 0.5]).numpy()
    target = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]).numpy()
    input_dict = {
        "margin": 5.0,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input1 = torch.tensor([1000.0, -1000.0, 0.001]).numpy()
    input2 = torch.tensor([999.9, -1000.1, -0.001]).numpy()
    target = torch.tensor([1.0, 1.0, -1.0]).numpy()
    input_dict = {
        "margin": 0.0,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (scalar)
    input1 = torch.tensor(-2.0).numpy()
    input2 = torch.tensor(2.0).numpy()
    target = torch.tensor(-1.0).numpy()
    input_dict = {
        "margin": 0.3,
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (reduction 'none' with reduce True)
    input1 = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).numpy()
    input2 = torch.tensor([-4.0, -1.0, -2.0, 0.5, 0.5, 1.5, 2.5, 3.5]).numpy()
    target = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]).numpy()
    input_dict = {
        "margin": 0.1,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (scalar float64)
    input1 = torch.tensor(3.14, dtype=torch.float64).numpy()
    input2 = torch.tensor(2.71, dtype=torch.float64).numpy()
    target = torch.tensor(1.0, dtype=torch.float64).numpy()
    input_dict = {
        "margin": 1.5,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MarginRankingLoss"] = margin_ranking_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MarginRankingLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MarginRankingLoss'.")


check_valid('torch.nn.MarginRankingLoss', generated_inputs['torch.nn.MarginRankingLoss'], lib="torch", suffix=0)
