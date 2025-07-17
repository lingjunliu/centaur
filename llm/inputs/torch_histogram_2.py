
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_histogram_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int bins and range
    input_tensor = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    bins_tensor = 5  # int
    range_val = (0.0, 5.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_val = False
    out_tensor = np.array([]) #Empty array
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_val, "weight": weight_tensor, "density": density_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 2:  bins as tensor, no range
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bins_tensor = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    range_val = None
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_val = False
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_val, "weight": weight_tensor, "density": density_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  with density=True
    input_tensor = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    bins_tensor = 5
    range_val = (0.0, 5.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_val = True
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_val, "weight": weight_tensor, "density": density_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: with weights
    input_tensor = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    bins_tensor = 5
    range_val = (0.0, 5.0)
    weight_tensor = np.array([2.0, 1.0, 3.0, 1.0, 2.0])
    density_val = False
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_val, "weight": weight_tensor, "density": density_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values in input
    input_tensor = np.array([-1.0, 2.0, -1.0, 3.0, -4.0])
    bins_tensor = 5
    range_val = (-5.0, 5.0)
    weight_tensor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    density_val = False
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_val, "weight": weight_tensor, "density": density_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Different weights
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bins_tensor = np.array([0.0, 2.0, 4.0, 6.0])
    range_val = None
    weight_tensor = np.array([0.5, 1.5, 0.5, 1.5, 0.5])
    density_val = True
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "bins": bins_tensor, "range": range_val, "weight": weight_tensor, "density": density_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.histogram_2"] = torch_histogram_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histogram_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histogram_2'.")

check_valid('torch.histogram', generated_inputs['torch.histogram_2'], lib="torch", suffix=2)
