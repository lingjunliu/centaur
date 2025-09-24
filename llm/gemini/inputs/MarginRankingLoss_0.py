
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def MarginRankingLoss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([4.0, 5.0, 6.0])
    target = np.array([1, -1, 1])
    input_dict = {
        "margin": 0.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([-1.0, -2.0, -3.0])
    input2 = np.array([-4.0, -5.0, -6.0])
    target = np.array([-1, 1, -1])
    input_dict = {
        "margin": 0.5,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([0.5, 1.5])
    input2 = np.array([1.0, 2.0])
    target = np.array([1, -1])
    input_dict = {
        "margin": 1.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([2.0])
    input2 = np.array([1.0])
    target = np.array([1])
    input_dict = {
        "margin": 0.2,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.5, 2.5, 3.5, 4.5])
    input2 = np.array([2.0, 3.0, 4.0, 5.0])
    target = np.array([1, 1, -1, -1])
    input_dict = {
        "margin": 1.5,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([0.0])
    input2 = np.array([0.0])
    target = np.array([1])
    input_dict = {
        "margin": 2.0,
        "size_average": False,
        "reduce": False,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MarginRankingLoss"] = MarginRankingLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MarginRankingLoss', generated_inputs['torch.nn.MarginRankingLoss'], lib="torch")
