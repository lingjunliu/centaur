
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    # Input 1 - basic case
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - different reduction type
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = False
    reduce = True
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - with negative values
    input = np.array([-0.1, 0.2, -0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - multi-dimensional tensor
    input = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    weight = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - different size_average
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = False
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - no reduce
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = False
    reduction = 'none'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - different reduction type
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = False
    reduce = False
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - with weights
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.5, 2.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - large values
    input = np.array([10.5, 20.3, 15.7], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - mixed values
    input = np.array([0.5, 0.2, 0.8], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy_with_logits_1"] = binary_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.binary_cross_entropy_with_logits_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy_with_logits_1'.")


check_valid('torch.nn.functional.binary_cross_entropy_with_logits', generated_inputs['torch.nn.functional.binary_cross_entropy_with_logits_1'], lib="torch", suffix=1)
