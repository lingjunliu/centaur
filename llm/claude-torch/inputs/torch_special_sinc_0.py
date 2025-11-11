
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sinc_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([0.0, 1.0, 2.0, 3.0])
    out = np.empty(4)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    out = np.empty(7)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([[0.5, 1.5], [2.5, 3.5]])
    out = np.empty((2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    out = np.empty((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array(1.5)
    out = np.empty(())
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([10.0, 20.0, 30.0, 40.0])
    out = np.empty(4)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([0.01, 0.1, 0.5, 0.9])
    out = np.empty(4)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([[-1.0, -0.5], [0.5, 1.0]])
    out = np.empty((2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.linspace(-5.0, 5.0, 20)
    out = np.empty(20)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.ones((2, 2, 2, 2)) * 0.5
    out = np.empty((2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.special.sinc"] = sinc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.sinc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.sinc'.")


check_valid('torch.special.sinc', generated_inputs['torch.special.sinc'], lib="torch", suffix=0)
