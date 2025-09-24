
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def cosine_embedding_loss_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    input2 = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    target = np.array([1], dtype=np.int8)
    margin = 0.5
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input1 = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    input2 = np.array([[-4.0, -5.0, -6.0]], dtype=np.float32)
    target = np.array([-1], dtype=np.int8)
    margin = 0.5
    size_average = False
    reduce = False
    reduction = 'none'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    target = np.array([1, -1], dtype=np.int8)
    margin = 0.2
    size_average = True
    reduce = False
    reduction = 'sum'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    input2 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array([1], dtype=np.int8)
    margin = 0.8
    size_average = False
    reduce = True
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input1 = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    input2 = np.array([[-4.0, 5.0, -6.0]], dtype=np.float32)
    target = np.array([-1], dtype=np.int8)
    margin = 0.0
    size_average = True
    reduce = False
    reduction = 'none'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input1 = np.array([[0.1, 0.2]], dtype=np.float32)
    input2 = np.array([[0.5, 0.6]], dtype=np.float32)
    target = np.array([1], dtype=np.int8)
    margin = -0.1
    size_average = False
    reduce = True
    reduction = 'sum'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input1 = np.array([[0.5]], dtype=np.float32)
    input2 = np.array([[0.7]], dtype=np.float32)
    target = np.array([1], dtype=np.int8)
    margin = 0.3
    size_average = True
    reduce = True
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input1 = np.array([[-1.0]], dtype=np.float32)
    input2 = np.array([[-0.5]], dtype=np.float32)
    target = np.array([-1], dtype=np.int8)
    margin = 0.9
    size_average = False
    reduce = False
    reduction = 'none'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input1 = np.array([[1.0, 2.0]], dtype=np.float32)
    input2 = np.array([[1.0, 2.0]], dtype=np.float32)
    target = np.array([1], dtype=np.int8)
    margin = 0.4
    size_average = True
    reduce = False
    reduction = 'sum'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input1 = np.array([[-1.0, -2.0]], dtype=np.float32)
    input2 = np.array([[-1.0, -2.0]], dtype=np.float32)
    target = np.array([-1], dtype=np.int8)
    margin = 0.6
    size_average = False
    reduce = True
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    target = np.array([1,1], dtype=np.int8)
    margin = 0.7
    size_average = True
    reduce = True
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    target = np.array([-1,-1], dtype=np.int8)
    margin = 0.3
    size_average = False
    reduce = False
    reduction = 'none'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.cosine_embedding_loss"] = cosine_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.cosine_embedding_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cosine_embedding_loss'.")

check_valid('torch.nn.functional.cosine_embedding_loss', generated_inputs['torch.nn.functional.cosine_embedding_loss'], lib="torch", suffix=0)
