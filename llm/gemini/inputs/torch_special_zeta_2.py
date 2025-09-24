
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def zeta_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    q = 1.0
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x with different dtype
    x = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    q = 1.5
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x with different shape
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    q = 2.0
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger x values
    x = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    q = 0.5
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x close to 1 (but > 1 to avoid pole for q=1)
    x = np.array([1.1, 1.2, 1.3], dtype=np.float64)
    q = 2.5
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different q value
    x = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    q = 0.1
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different q value
    x = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    q = 10.0
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional x and different q
    x = np.array([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]], dtype=np.float64)
    q = 3.0
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x with single element
    x = np.array([5.0], dtype=np.float64)
    q = 0.75
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: x with larger shape and q close to zero.
    x = np.ones((2,2,2), dtype=np.float64) * 3
    q = 0.01
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: x with different dtype and q close to 1
    x = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    q = 0.9
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: x with large values and small q
    x = np.array([100.0, 110.0, 120.0], dtype=np.float64)
    q = 0.001
    input_dict = {"x": x, "q": q}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.zeta_2"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.zeta_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.zeta_2'.")

check_valid('torch.special.zeta', generated_inputs['torch.special.zeta_2'], lib="torch", suffix=2)
