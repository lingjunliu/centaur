
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1: 1D
    input = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    size_average = np.bool_(True)
    reduce = np.bool_(True)
    reduction = 'mean'
    pos_weight = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 2: 2D (N=2, C=3)
    input = np.array([[0.2, -0.3, 1.5],
                      [2.0, -1.0, 0.0]], dtype=np.float64)
    target = np.clip(np.array([[0.0, 1.0, 0.7],
                               [1.0, 0.0, 0.5]], dtype=np.float64), 0.0, 1.0)
    weight = np.array([[1.0, 2.0, 0.5],
                       [0.8, 1.2, 1.0]], dtype=np.float64)
    size_average = np.bool_(False)
    reduce = np.bool_(True)
    reduction = 'sum'
    pos_weight = np.array(1.3, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 3: 3D (N=2, C=3, L=5)
    input = np.linspace(-2, 2, num=2*3*5, dtype=np.float32).reshape(2, 3, 5)
    target = np.clip(np.linspace(0, 1, num=2*3*5, dtype=np.float32).reshape(2, 3, 5), 0.0, 1.0)
    weight = np.ones((2, 3, 5), dtype=np.float32)
    size_average = np.bool_(True)
    reduce = np.bool_(True)
    reduction = 'mean'
    pos_weight = np.array(0.9, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 4: 4D (N=1, C=2, H=2, W=2)
    input = np.array([[[[0.5, -0.5],
                        [1.0, -1.0]],
                       [[2.0, -2.0],
                        [0.3, -0.3]]]], dtype=np.float32)
    target = np.clip(np.array([[[[1.0, 0.0],
                                 [1.0, 0.0]],
                                [[0.0, 1.0],
                                 [0.2, 0.8]]]], dtype=np.float32), 0.0, 1.0)
    weight = np.array([[[[1.0, 1.0],
                         [1.0, 1.0]],
                        [[0.5, 0.5],
                         [0.5, 0.5]]]], dtype=np.float32)
    size_average = np.bool_(True)
    reduce = np.bool_(True)
    reduction = 'mean'
    pos_weight = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 5: scalar
    input = np.array(0.7, dtype=np.float32)
    target = np.array(1.0, dtype=np.float32)
    weight = np.array(2.0, dtype=np.float32)
    size_average = np.bool_(True)
    reduce = np.bool_(True)
    reduction = 'mean'
    pos_weight = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 6: 4D (N=2, C=1, H=4, W=3)
    input = np.linspace(-1.5, 1.5, num=2*1*4*3, dtype=np.float64).reshape(2, 1, 4, 3)
    target = np.clip(np.linspace(0.0, 1.0, num=2*1*4*3, dtype=np.float64).reshape(2, 1, 4, 3), 0.0, 1.0)
    weight = np.ones((2, 1, 4, 3), dtype=np.float64) * 0.7
    size_average = np.bool_(False)
    reduce = np.bool_(True)
    reduction = 'sum'
    pos_weight = np.array(2.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 7: 3D (N=5, C=1, L=7)
    input = (np.arange(5*1*7, dtype=np.float32).reshape(5, 1, 7) - 10.0) / 3.0
    target = np.clip(np.random.RandomState(0).rand(5, 1, 7).astype(np.float32), 0.0, 1.0)
    weight = np.linspace(0.5, 1.5, num=5*1*7, dtype=np.float32).reshape(5, 1, 7)
    size_average = np.bool_(True)
    reduce = np.bool_(False)
    reduction = 'none'
    pos_weight = np.array(1.1, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 8: 2D (N=8, C=1)
    input = np.linspace(-3.0, 3.0, num=8*1, dtype=np.float16).reshape(8, 1)
    target = np.clip(np.linspace(0.0, 1.0, num=8*1, dtype=np.float16).reshape(8, 1), 0.0, 1.0)
    weight = np.full((8, 1), 0.9, dtype=np.float16)
    size_average = np.bool_(True)
    reduce = np.bool_(True)
    reduction = 'mean'
    pos_weight = np.array(0.7, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 9: 4D (N=2, C=3, H=1, W=1)
    input = ((np.arange(2*3*1*1, dtype=np.float32).reshape(2, 3, 1, 1) - 3.0) / 2.0)
    target = np.clip(np.array([[[[0.0]], [[1.0]], [[0.5]]],
                               [[[1.0]], [[0.0]], [[0.2]]]], dtype=np.float32), 0.0, 1.0)
    weight = np.ones((2, 3, 1, 1), dtype=np.float32) * 0.5
    size_average = np.bool_(False)
    reduce = np.bool_(True)
    reduction = 'sum'
    pos_weight = np.array(1.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

    # Input 10: 4D (N=2, C=4, H=2, W=3)
    input = np.linspace(-1.0, 1.0, num=2*4*2*3, dtype=np.float32).reshape(2, 4, 2, 3)
    target = np.clip(np.linspace(0.0, 1.0, num=2*4*2*3, dtype=np.float32).reshape(2, 4, 2, 3), 0.0, 1.0)
    weight = np.full((2, 4, 2, 3), 0.3, dtype=np.float32)
    size_average = np.bool_(True)
    reduce = np.bool_(False)
    reduction = 'none'
    pos_weight = np.array(0.6, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        "input": input, "target": target, "weight": weight,
        "size_average": size_average, "reduce": reduce, "reduction": reduction,
        "pos_weight": pos_weight
    }))

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
