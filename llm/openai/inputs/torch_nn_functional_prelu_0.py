
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def prelu_inputs():
    list_of_inputs = []
    
    input = np.array([-1.0, 0.0, 1.0, 2.5, -3.5], dtype=np.float32)
    weight = np.array(0.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = np.linspace(-2, 2, 5, dtype=np.float64)
    weight = np.array([0.1], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = np.array(-1.5, dtype=np.float32)
    weight = np.array([0.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = (np.arange(12, dtype=np.float32).reshape(4, 3) - 6.0)
    weight = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = np.empty((0, 2), dtype=np.float32)
    weight = np.array([0.3, -0.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = (np.arange(70, dtype=np.float32) - 10.0).reshape(2, 5, 7).astype(np.float16)
    weight = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = ((np.arange(3 * 4 * 8 * 8, dtype=np.float32).reshape(3, 4, 8, 8) - 500.0) / 50.0)
    weight = np.array([-0.1, 0.0, 0.1, 0.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = (np.arange(1 * 2 * 3 * 4 * 5, dtype=np.float64).reshape(1, 2, 3, 4, 5) - 10.0)
    weight = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = (np.arange(2 * 3 * 1 * 2 * 1 * 2, dtype=np.float32).reshape(2, 3, 1, 2, 1, 2) - 1.0)
    weight = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = np.array([[-1.0], [0.0], [1.0], [2.0], [-2.0]], dtype=np.float32)
    weight = np.array([0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = (np.arange(15, dtype=np.float32).reshape(5, 3) - 7.0)
    weight = np.array(0.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = np.linspace(-1.0, 1.0, 20, dtype=np.float64).reshape(5, 1, 4)
    weight = np.array([0.25], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = (np.arange(16, dtype=np.float16) - 8).reshape(2, 2, 2, 2)
    weight = np.array(0.1, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = np.array(0.0, dtype=np.float64)
    weight = np.array(-0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    input = np.array([], dtype=np.float32)
    weight = np.array([0.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight}))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.prelu"] = prelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.prelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.prelu'.")


check_valid('torch.nn.functional.prelu', generated_inputs['torch.nn.functional.prelu'], lib="torch", suffix=0)
