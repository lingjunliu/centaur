
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def huber_loss_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    delta1 = 1.0
    reduction1 = 'mean'
    input_dict1 = {"input": input1, "target": target1, "delta": delta1, "reduction": reduction1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensors with different values
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.2, 2.3], [3.4, 4.5]], dtype=np.float32)
    delta2 = 0.5
    reduction2 = 'sum'
    input_dict2 = {"input": input2, "target": target2, "delta": delta2, "reduction": reduction2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values and different delta
    input3 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    target3 = np.array([-1.5, -2.5, -3.5], dtype=np.float32)
    delta3 = 2.0
    reduction3 = 'none'
    input_dict3 = {"input": input3, "target": target3, "delta": delta3, "reduction": reduction3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger tensors
    input4 = np.random.rand(10, 10).astype(np.float32)
    target4 = np.random.rand(10, 10).astype(np.float32)
    delta4 = 0.75
    reduction4 = 'mean'
    input_dict4 = {"input": input4, "target": target4, "delta": delta4, "reduction": reduction4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Delta close to zero
    input5 = np.array([1.0, 2.0], dtype=np.float32)
    target5 = np.array([1.01, 2.02], dtype=np.float32)
    delta5 = 0.01
    reduction5 = 'sum'
    input_dict5 = {"input": input5, "target": target5, "delta": delta5, "reduction": reduction5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensors
    input6 = np.random.rand(3, 4, 5).astype(np.float32)
    target6 = np.random.rand(3, 4, 5).astype(np.float32)
    delta6 = 1.5
    reduction6 = 'mean'
    input_dict6 = {"input": input6, "target": target6, "delta": delta6, "reduction": reduction6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Delta > 1
    input7 = np.array([2.0, 4.0], dtype=np.float32)
    target7 = np.array([1.0, 3.0], dtype=np.float32)
    delta7 = 3.0
    reduction7 = 'sum'
    input_dict7 = {"input": input7, "target": target7, "delta": delta7, "reduction": reduction7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Identical tensors
    input8 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target8 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    delta8 = 1.0
    reduction8 = 'none'
    input_dict8 = {"input": input8, "target": target8, "delta": delta8, "reduction": reduction8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Large differences
    input9 = np.array([0.0, 0.0], dtype=np.float32)
    target9 = np.array([10.0, 20.0], dtype=np.float32)
    delta9 = 5.0
    reduction9 = 'mean'
    input_dict9 = {"input": input9, "target": target9, "delta": delta9, "reduction": reduction9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Single element tensors
    input10 = np.array([1.0], dtype=np.float32)
    target10 = np.array([1.5], dtype=np.float32)
    delta10 = 0.5
    reduction10 = 'sum'
    input_dict10 = {"input": input10, "target": target10, "delta": delta10, "reduction": reduction10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Another set of random numbers
    input11 = np.random.rand(5, 5).astype(np.float32)
    target11 = np.random.rand(5, 5).astype(np.float32)
    delta11 = 2.5
    reduction11 = 'mean'
    input_dict11 = {"input": input11, "target": target11, "delta": delta11, "reduction": reduction11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.huber_loss"] = huber_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.huber_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.huber_loss'.")

check_valid('torch.nn.functional.huber_loss', generated_inputs['torch.nn.functional.huber_loss'], lib="torch", suffix=0)
