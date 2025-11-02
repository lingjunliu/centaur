
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = np.array([0.5, -0.3, 0.8]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = np.array([[0.1, -0.5], [0.3, -0.2]]).astype(np.float32)
    target = np.array([[1.0, 0.0], [1.0, 0.0]]).astype(np.float32)
    weight = np.array([[1.0, 1.0], [1.0, 1.0]]).astype(np.float32)
    size_average = False
    reduce = True
    reduction = 'sum'
    pos_weight = np.array([1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = np.array([0.1]).astype(np.float32)
    target = np.array([1.0]).astype(np.float32)
    weight = np.array([1.0]).astype(np.float32)
    size_average = True
    reduce = False
    reduction = 'none'
    pos_weight = np.array([1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = np.array([-0.1, 0.5, -0.2, 0.7]).astype(np.float32)
    target = np.array([0.0, 1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([0.5, 0.5, 0.5, 0.5]).astype(np.float32)
    size_average = False
    reduce = True
    reduction = 'mean'
    pos_weight = np.array([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = np.array([0.0]).astype(np.float32)
    target = np.array([0.0]).astype(np.float32)
    weight = np.array([1.0]).astype(np.float32)
    size_average = True
    reduce = False
    reduction = 'none'
    pos_weight = np.array([1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = np.array([0.5, -0.5, 0.8]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = False
    reduce = False
    reduction = 'none'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = np.array([[0.1, -0.5], [0.3, -0.2]]).astype(np.float32)
    target = np.array([[1.0, 0.0], [1.0, 0.0]]).astype(np.float32)
    weight = np.array([[1.0, 1.0], [1.0, 1.0]]).astype(np.float32)
    size_average = True
    reduce = False
    reduction = 'sum'
    pos_weight = np.array([1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = np.array([0.1, -0.5, 0.3]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = False
    reduce = True
    reduction = 'mean'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = np.array([0.1]).astype(np.float32)
    target = np.array([1.0]).astype(np.float32)
    weight = np.array([1.0]).astype(np.float32)
    size_average = True
    reduce = True
    reduction = 'sum'
    pos_weight = np.array([1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = np.array([0.5, -0.5, 0.8]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = True
    reduce = True
    reduction = 'none'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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
