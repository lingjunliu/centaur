
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def triplet_margin_loss_inputs():
    list_of_inputs = []

    # Input 1
    anchor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positive = np.array([1.1, 2.1, 3.1], dtype=np.float32)
    negative = np.array([1.2, 2.2, 3.2], dtype=np.float32)
    margin = 1.0
    p = 2.0
    eps = 1e-6
    swap = False
    reduction = 'mean'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    anchor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    positive = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)
    negative = np.array([[1.2, 2.2], [3.2, 4.2]], dtype=np.float32)
    margin = 0.5
    p = 1.0
    eps = 1e-8
    swap = True
    reduction = 'sum'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    anchor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    positive = np.array([[[1.1, 2.1], [3.1, 4.1]], [[5.1, 6.1], [7.1, 8.1]]], dtype=np.float32)
    negative = np.array([[[1.2, 2.2], [3.2, 4.2]], [[5.2, 6.2], [7.2, 8.2]]], dtype=np.float32)
    margin = 0.2
    p = 3.0
    eps = 1e-5
    swap = False
    reduction = 'none'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    anchor = np.array([1.0], dtype=np.float32)
    positive = np.array([1.1], dtype=np.float32)
    negative = np.array([1.2], dtype=np.float32)
    margin = 1.5
    p = 0.5
    eps = 1e-7
    swap = True
    reduction = 'mean'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    anchor = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    positive = np.array([-1.1, -2.1, -3.1], dtype=np.float32)
    negative = np.array([-1.2, -2.2, -3.2], dtype=np.float32)
    margin = 2.0
    p = 2.5
    eps = 1e-9
    swap = False
    reduction = 'sum'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    anchor = np.random.rand(5, 5).astype(np.float32)
    positive = np.random.rand(5, 5).astype(np.float32)
    negative = np.random.rand(5, 5).astype(np.float32)
    margin = 0.75
    p = 1.5
    eps = 1e-4
    swap = True
    reduction = 'none'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Removed invalid input (margin=0)
    
    # Input 8
    anchor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    positive = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    negative = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    margin = 0.25
    p = 2.0
    eps = 1e-6
    swap = True
    reduction = 'mean'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    anchor = np.random.randn(2, 3, 4).astype(np.float32)
    positive = np.random.randn(2, 3, 4).astype(np.float32)
    negative = np.random.randn(2, 3, 4).astype(np.float32)
    margin = 1.0
    p = 2.0
    eps = 1e-6
    swap = False
    reduction = 'sum'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    anchor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    positive = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float32)
    negative = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    margin = 1.0
    p = 2.0
    eps = 1e-6
    swap = True
    reduction = 'none'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    anchor = np.array([1.0, 2.0], dtype=np.float32)
    positive = np.array([3.0, 4.0], dtype=np.float32)
    negative = np.array([5.0, 6.0], dtype=np.float32)
    margin = 0.1
    p = 2.0
    eps = 1e-6
    swap = True
    reduction = 'mean'

    input_dict = {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "margin": margin,
        "p": p,
        "eps": eps,
        "swap": swap,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.triplet_margin_loss"] = triplet_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.triplet_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.triplet_margin_loss'.")

check_valid('torch.nn.functional.triplet_margin_loss', generated_inputs['torch.nn.functional.triplet_margin_loss'], lib="torch", suffix=0)
