
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def huber_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.1, 1.9, 3.2], dtype=np.float32)
    input_dict1 = {"reduction": "mean", "delta": 1.0, "input": input1, "target": target1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.2, 1.8], [2.9, 4.1]], dtype=np.float32)
    input_dict2 = {"reduction": "sum", "delta": 0.5, "input": input2, "target": target2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    target3 = np.array([-1.1, -2.1, -2.9], dtype=np.float32)
    input_dict3 = {"reduction": "none", "delta": 2.0, "input": input3, "target": target3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    target4 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict4 = {"reduction": "mean", "delta": 0.1, "input": input4, "target": target4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    target5 = np.array([5.0, 15.0, 25.0], dtype=np.float32)
    input_dict5 = {"reduction": "sum", "delta": 5.0, "input": input5, "target": target5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0], dtype=np.float32)
    target6 = np.array([2.0], dtype=np.float32)
    input_dict6 = {"reduction": "mean", "delta": 1.0, "input": input6, "target": target6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 4, 4).astype(np.float32)
    target7 = np.random.rand(4, 4, 4).astype(np.float32)
    input_dict7 = {"reduction": "none", "delta": 1.5, "input": input7, "target": target7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    target8 = np.array([0.0, -1.0, 4.0], dtype=np.float32)
    input_dict8 = {"reduction": "mean", "delta": 0.2, "input": input8, "target": target8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target9 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict9 = {"reduction": "sum", "delta": 10.0, "input": input9, "target": target9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([0.5, -0.5, 1.5], dtype=np.float32)
    target10 = np.array([0.6, -0.4, 1.4], dtype=np.float32)
    input_dict10 = {"reduction": "mean", "delta": 0.3, "input": input10, "target": target10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.HuberLoss"] = huber_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.HuberLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.HuberLoss'.")


check_valid('torch.nn.HuberLoss', generated_inputs['torch.nn.HuberLoss'], lib="torch", suffix=0)
