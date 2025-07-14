
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def histogram_inputs():
    list_of_inputs = []

    # Input 1: Basic example with integer bins
    input_tensor = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    bins_tensor = 4
    range_tuple = (0.0, 5.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Bins as a tensor
    input_tensor = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    bins_tensor = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    range_tuple = (0.0, 5.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_bool = True
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different weights
    input_tensor = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    bins_tensor = 4
    range_tuple = (0.0, 5.0)
    weight_tensor = np.array([2.0, 1.0, 3.0, 1.0, 2.0])
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    input_tensor = np.array([-1.0, -2.0, -1. 0, -3.0, -4.0])
    bins_tensor = 4
    range_tuple = (-5.0, 0.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_bool = True
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero range
    input_tensor = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    bins_tensor = 4
    range_tuple = (0.0, 1.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different range
    input_tensor = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    bins_tensor = np.array([1.0, 4.0])
    range_tuple = (0.0, 5.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Edge case: empty input
    input_tensor = np.array([])
    bins_tensor = 5
    range_tuple = (0.0, 5.0)
    weight_tensor = np.array([])
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Edge case: empty weights
    input_tensor = np.array([1.0, 2.0, 3.0])
    bins_tensor = 3
    range_tuple = (0.0, 3.0)
    weight_tensor = np.array([])
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different data type for input
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    bins_tensor = 3
    range_tuple = (0.0, 3.0)
    weight_tensor = np.array([1.0, 1.0, 1.0])
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Weight with different dtype
    input_tensor = np.array([1.0, 2.0, 3.0])
    bins_tensor = 3
    range_tuple = (0.0, 3.0)
    weight_tensor = np.array([1, 1, 1], dtype=np.int32)
    density_bool = False
    out_tensor = None
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_tuple, "weight": weight_tensor, "density": density_bool, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.histogram_2"] = histogram_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histogram_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histogram_2'.")

check_valid('torch.histogram', generated_inputs['torch.histogram_2'], lib="torch", suffix=2)
