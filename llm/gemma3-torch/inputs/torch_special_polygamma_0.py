
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def polygamma_inputs():
    list_of_inputs = []
    
    input1 = np.array(0, dtype=np.int64)
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    input_dict1 = {
        "n": input1,
        "x": torch.tensor(x1).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array(1, dtype=np.int64)
    x2 = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    
    input_dict2 = {
        "n": input2,
        "x": torch.tensor(x2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array(2, dtype=np.int64)
    x3 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    
    input_dict3 = {
        "n": input3,
        "x": torch.tensor(x3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array(3, dtype=np.int64)
    x4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    
    input_dict4 = {
        "n": input4,
        "x": torch.tensor(x4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array(0, dtype=np.int64)
    x5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    
    input_dict5 = {
        "n": input5,
        "x": torch.tensor(x5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array(1, dtype=np.int64)
    x6 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    
    input_dict6 = {
        "n": input6,
        "x": torch.tensor(x6).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    
    input7 = np.array(4, dtype=np.int64)
    x7 = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    
    input_dict7 = {
        "n": input7,
        "x": torch.tensor(x7).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array(0, dtype=np.int64)
    x8 = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    
    input_dict8 = {
        "n": input8,
        "x": torch.tensor(x8).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs["torch.special.polygamma"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.polygamma'.")


check_valid('torch.special.polygamma', generated_inputs['torch.special.polygamma'], lib="torch", suffix=0)
