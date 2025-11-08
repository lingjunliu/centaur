
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    input1 = np.array([[-1.0, 0.5, 2.0]])
    target1 = np.array([[0.0, 1.0, 0.0]])
    weight1 = np.array([0.2, 0.5, 0.3])
    size_average1 = True
    reduce1 = True
    pos_weight1 = np.array([1.0])
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "size_average": size_average1,
        "reduce": reduce1,
        "pos_weight": pos_weight1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[0.1, -0.8], [1.2, 0.3]])
    target2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    weight2 = np.array([1.0, 1.0])
    size_average2 = False
    reduce2 = True
    pos_weight2 = np.array([2.0])
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "size_average": size_average2,
        "reduce": reduce2,
        "pos_weight": pos_weight2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-2.0, 1.5, -0.7, 3.1]])
    target3 = np.array([[0.0, 1.0, 0.0, 1.0]])
    weight3 = np.array([0.8, 0.2, 0.5, 0.9])
    size_average3 = True
    reduce3 = True
    pos_weight3 = np.array([1.0])
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": weight3,
        "size_average": size_average3,
        "reduce": reduce3,
        "pos_weight": pos_weight3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-0.5, 0.2, -1.0, 1.8, 0.7]])
    target4 = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]])
    weight4 = np.array([0.3, 0.7, 0.4, 0.6, 0.5])
    size_average4 = False
    reduce4 = True
    pos_weight4 = np.array([1.0])

    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "size_average": size_average4,
        "reduce": reduce4,
        "pos_weight": pos_weight4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy_with_logits"] = binary_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.binary_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy_with_logits'.")


check_valid('torch.nn.functional.binary_cross_entropy_with_logits', generated_inputs['torch.nn.functional.binary_cross_entropy_with_logits'], lib="torch", suffix=0)
