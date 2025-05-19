
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
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

    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    target = np.array([[1, -1], [1,1]], dtype=np.int32)
    margin = 0.2
    size_average = False
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

    input1 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    input2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    target = np.array([-1, -1, 1, 1], dtype=np.int32)
    margin = 1.0
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

    input1 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input2 = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    target = np.array([[1, -1], [1,1]], dtype=np.int32)
    margin = -0.5
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

    input1 = np.array([1.0], dtype=np.float32)
    input2 = np.array([2.0], dtype=np.float32)
    target = np.array([1], dtype=np.int32)
    margin = 0.0
    size_average = True
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

    return list_of_inputs

generated_inputs = margin_ranking_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MarginRankingLoss', generated_inputs)
