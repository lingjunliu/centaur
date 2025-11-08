
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def smooth_l1_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict1 = {
        "beta": 1.0,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict2 = {
        "beta": 0.5,
        "reduction": "sum",
        "size_average": False,
        "reduce": False,
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    target3 = np.array([1.0, -2.0], dtype=np.float32)
    input_dict3 = {
        "beta": 2.0,
        "reduction": "none",
        "size_average": True,
        "reduce": True,
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0], dtype=np.float32)
    target4 = np.array([2.0], dtype=np.float32)
    input_dict4 = {
        "beta": 0.1,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    target5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict5 = {
        "beta": 1.0,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target6 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict6 = {
        "beta": 5.0,
        "reduction": "sum",
        "size_average": False,
        "reduce": True,
        "input": input6,
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.0], dtype=np.float32)
    target7 = np.array([0.0], dtype=np.float32)
    input_dict7 = {
        "beta": 0.0,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input7,
        "target": target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    target8 = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    input_dict8 = {
        "beta": 10.0,
        "reduction": "none",
        "size_average": False,
        "reduce": False,
        "input": input8,
        "target": target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0], dtype=np.float32)
    target9 = np.array([3.0, 4.0], dtype=np.float32)
    input_dict9 = {
        "beta": 1.5,
        "reduction": "mean",
        "size_average": True,
        "reduce": True,
        "input": input9,
        "target": target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[-1.0], [2.0]], dtype=np.float32)
    target10 = np.array([[-1.0], [2.0]], dtype=np.float32)
    input_dict10 = {
        "beta": 0.5,
        "reduction": "sum",
        "size_average": False,
        "reduce": True,
        "input": input10,
        "target": target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smooth_l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.SmoothL1Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SmoothL1Loss'.")


check_valid('torch.nn.SmoothL1Loss', generated_inputs['torch.nn.SmoothL1Loss'], lib="torch", suffix=0)
