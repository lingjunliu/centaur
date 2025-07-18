
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy

def broadcast_tensors_inputs():
    list_of_inputs = []

    # Input 1: Two 1D tensors of the same shape
    input_dict_1 = {
        'tensors': numpy.stack([
            torch.tensor([1, 2, 3]).numpy(),
            torch.tensor([4, 5, 6]).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Three 2D tensors of the same shape, float
    input_dict_2 = {
        'tensors': numpy.stack([
            torch.ones(2, 3, dtype=torch.float32).numpy(),
            torch.zeros(2, 3, dtype=torch.float32).numpy(),
            torch.full((2, 3), 5, dtype=torch.float32).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: A single 3D tensor, stacked
    input_dict_3 = {
        'tensors': numpy.stack([
            torch.randn(2, 4, 5).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tensors with negative values
    input_dict_4 = {
        'tensors': numpy.stack([
            torch.full((2, 2), -1.5).numpy(),
            torch.full((2, 2), -3.0).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Different compatible data types (int, float)
    input_dict_5 = {
        'tensors': numpy.stack([
            torch.ones((3, 2), dtype=torch.int32).numpy(),
            torch.zeros((3, 2), dtype=torch.float32).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Higher-dimensional tensors (4D)
    input_dict_6 = {
        'tensors': numpy.stack([
            torch.ones(2, 1, 3, 4, dtype=torch.int64).numpy(),
            torch.zeros(2, 1, 3, 4, dtype=torch.int64).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Scalar (0-D) tensors
    input_dict_7 = {
        'tensors': numpy.stack([
            torch.tensor(5).numpy(),
            torch.tensor(-10).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensors with a dimension of size 0
    input_dict_8 = {
        'tensors': numpy.stack([
            torch.empty(2, 0).numpy(),
            torch.empty(2, 0).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensors with large values
    input_dict_9 = {
        'tensors': numpy.stack([
            torch.full((3, 3), 1e9, dtype=torch.float64).numpy(),
            torch.full((3, 3), -1e9, dtype=torch.float64).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Four 1D tensors of the same shape
    input_dict_10 = {
        'tensors': numpy.stack([
            torch.ones((5,)).numpy(),
            torch.zeros((5,)).numpy(),
            torch.full((5,), 2).numpy(),
            torch.full((5,), -2).numpy()
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: An empty list of tensors, represented by a (0, ...) shaped array
    input_dict_11 = {
        'tensors': numpy.empty((0, 2, 2), dtype=numpy.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Actual broadcasting case
    input_dict_12 = {
        'tensors': [
            torch.randn(3, 1).numpy(),
            torch.randn(1, 4).numpy()
        ]
    }
    # This input is removed as the testing framework seems to require a stackable list of tensors.
    # The framework expects a single numpy array for the 'tensors' argument.
    
    return list_of_inputs

generated_inputs["torch.broadcast_tensors"] = broadcast_tensors_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.broadcast_tensors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.broadcast_tensors'.")

check_valid('torch.broadcast_tensors', generated_inputs['torch.broadcast_tensors'], lib="torch", suffix=0)
