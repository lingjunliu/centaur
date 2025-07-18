
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def multigammaln_inputs():
    list_of_inputs = []
    
    # Condition for validity: input > (p-1)/2

    # Input 1: Basic 1D float32 tensor
    # p=2 -> input > 0.5. All values are valid.
    input_dict_1 = {
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'p': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float64 tensor with a larger p
    # p=5 -> input > 2.0. All values are valid.
    input_dict_2 = {
        'input': np.array([[3.5, 4.5], [5.5, 6.5]], dtype=np.float64),
        'p': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: p=1, which is equivalent to lgamma
    # p=1 -> input > 0. All values are valid.
    input_dict_3 = {
        'input': np.array([0.5, 1.5, 10.0], dtype=np.float32),
        'p': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: A valid input case
    # p=4 -> input > 1.5. All values are valid.
    input_dict_4 = {
        'input': np.array([1.6, 2.0, 3.0], dtype=np.float32),
        'p': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Simple valid case
    # p=3 -> input > 1.0. All values are valid.
    input_dict_5 = {
        'input': np.array([2.0, 3.0, 4.0], dtype=np.float32),
        'p': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D tensor
    # p=2 -> input > 0.5. All values are valid.
    input_dict_6 = {
        'input': np.arange(2.1, 10.1, 1.0, dtype=np.float32).reshape((2, 2, 2)),
        'p': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Scalar input (0-D tensor)
    # p=6 -> input > 2.5. 5.0 is valid.
    input_dict_7 = {
        'input': np.array(5.0, dtype=np.float32),
        'p': 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Values are very close to the boundary condition (p-1)/2
    # p=7 -> input > 3.0. All values are valid.
    input_dict_8 = {
        'input': np.array([3.0001, 3.1, 4.0], dtype=np.float64),
        'p': 7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large p value
    # p=15 -> input > 7.0. All values are valid.
    input_dict_9 = {
        'input': np.array([10.0, 11.0, 12.0], dtype=np.float32),
        'p': 15
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Another 2D valid case
    # p=6 -> input > 2.5. All values are valid.
    input_dict_10 = {
        'input': np.array([[2.6, 3.0], [3.5, 4.0]], dtype=np.float32),
        'p': 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Empty tensor
    input_dict_11 = {
        'input': np.array([], dtype=np.float32).reshape(0, 2),
        'p': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["torch.special.multigammaln"] = multigammaln_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.multigammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.multigammaln'.")

check_valid('torch.special.multigammaln', generated_inputs['torch.special.multigammaln'], lib="torch", suffix=0)
