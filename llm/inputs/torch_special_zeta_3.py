
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def special_zeta_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 1D tensors, x > 1, q > 0
    input_dict_1 = {
        'x': torch.tensor([2.0, 3.0, 5.0], dtype=torch.float32).numpy(),
        'q': torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Broadcasting - scalar x, 2D q
    input_dict_2 = {
        'x': torch.tensor(4.0, dtype=torch.float64).numpy(),
        'q': torch.tensor([[1.1, 2.2], [3.3, 4.4]], dtype=torch.float64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Broadcasting - 2D x, scalar q
    input_dict_3 = {
        'x': torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32).numpy(),
        'q': torch.tensor(0.7, dtype=torch.float32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: More complex broadcasting
    input_dict_4 = {
        'x': torch.tensor([[2.0], [3.0]], dtype=torch.float64).numpy(),
        'q': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: x values <= 1 (analytic continuation), q > 0
    input_dict_5 = {
        'x': torch.tensor([-1.5, 0.0, 0.5, 1.0], dtype=torch.float32).numpy(),
        'q': torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: q values close to 0 (but still positive)
    input_dict_6 = {
        'x': torch.tensor([2.0, 2.0], dtype=torch.float64).numpy(),
        'q': torch.tensor([1e-6, 1e-3], dtype=torch.float64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Large values for x and q
    input_dict_7 = {
        'x': torch.tensor([100.0, 200.0], dtype=torch.float32).numpy(),
        'q': torch.tensor([50.0, 60.0], dtype=torch.float32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D x, 1D q (broadcasting)
    input_dict_8 = {
        'x': torch.arange(2.0, 10.0, 1.0, dtype=torch.float64).reshape(2, 2, 2).numpy(),
        'q': torch.tensor([1.0, 2.0], dtype=torch.float64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Empty tensors
    input_dict_9 = {
        'x': torch.tensor([], dtype=torch.float32).numpy(),
        'q': torch.tensor([], dtype=torch.float32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Broadcasting with different numbers of dimensions
    input_dict_10 = {
        'x': torch.tensor([[[1.1]], [[2.2]]], dtype=torch.float32).numpy(),
        'q': torch.arange(1.0, 7.0, 1.0, dtype=torch.float32).reshape(2, 3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.special.zeta_3"] = special_zeta_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.zeta_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.zeta_3'.")

check_valid('torch.special.zeta', generated_inputs['torch.special.zeta_3'], lib="torch", suffix=3)
