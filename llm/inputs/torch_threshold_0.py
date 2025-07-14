
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    threshold = 0.5
    value = 0.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    threshold = 1.0
    value = -1.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All negative values
    input = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    threshold = -2.0
    value = 0.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Threshold at zero
    input = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    threshold = 0.0
    value = 5.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Threshold larger than max value
    input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    threshold = 4.0
    value = 0.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Threshold smaller than min value
    input = np.array([-3.0, -2.0, -1.0], dtype=np.float32)
    threshold = -4.0
    value = 1.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero value
    input = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    threshold = 0.5
    value = 0.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor
    input = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    threshold = 5.0
    value = -1.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large threshold and value
    input = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    threshold = 250.0
    value = 1000.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Edge Case - Input values equal to threshold
    input = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    threshold = 1.0
    value = 0.0
    input_dict = {"input": input, "threshold": threshold, "value": value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.threshold'.")

check_valid('torch.threshold', generated_inputs['torch.threshold'], lib="torch", suffix=0)
