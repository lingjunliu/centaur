
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy

def get_torch_histogram_inputs():
    """
    Generates a list of valid inputs for the torch.histogram function.
    To satisfy the testing harness which requires `bins` to be a tensor-like object
    and `range` to be a tuple, `bins` is provided as a 0-dimensional numpy array.
    This is compatible with both the test harness and the torch.histogram API.
    """
    list_of_inputs = []
    # A placeholder for the 'out' tensor to satisfy the strict signature.
    out_tensor = torch.tensor([]).numpy()

    # Input 1: Basic case
    input_tensor = torch.tensor([1., 2., 1., 5.])
    input_1 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(5),
        'range': (0., 5.),
        'weight': torch.ones_like(input_tensor).numpy(),
        'density': False,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: With specific weights
    input_tensor = torch.tensor([1., 2., 1.])
    input_2 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(4),
        'range': (0., 3.),
        'weight': torch.tensor([0.5, 1.0, 2.0]).numpy(),
        'density': False,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: With density=True
    input_tensor = torch.tensor([0.1, 0.8, 0.9, 0.1, 0.4])
    input_3 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(2),
        'range': (0., 1.),
        'weight': torch.ones_like(input_tensor).numpy(),
        'density': True,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: 2D input tensor
    input_tensor = torch.tensor([[1., 2.], [3., 1.]])
    input_4 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(10),
        'range': (0., 4.),
        'weight': torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy(),
        'density': False,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Negative values in input
    input_tensor = torch.tensor([-1., -2., 0., 1., -1.5])
    input_5 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(3),
        'range': (-3., 1.5),
        'weight': torch.ones_like(input_tensor).numpy(),
        'density': True,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Large random input
    input_tensor = torch.randn(100)
    input_6 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(10),
        'range': (-4., 4.),
        'weight': torch.ones_like(input_tensor).numpy(),
        'density': False,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Float64 dtype
    input_tensor = torch.tensor([10.5, 20.1, 30.8], dtype=torch.float64)
    input_7 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(3),
        'range': (10., 40.),
        'weight': torch.tensor([1., 1., 2.], dtype=torch.float64).numpy(),
        'density': False,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Range outside of input values
    input_tensor = torch.tensor([1., 2., 3.])
    input_8 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(2),
        'range': (10., 30.),
        'weight': torch.ones_like(input_tensor).numpy(),
        'density': False,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Single bin
    input_tensor = torch.tensor([1., 2., 3., 4., 5.])
    input_9 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(1),
        'range': (0., 10.),
        'weight': torch.ones_like(input_tensor).numpy(),
        'density': False,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: 3D input tensor
    input_tensor = torch.rand((2, 3, 4))
    input_10 = {
        'input': input_tensor.numpy(),
        'bins': numpy.array(4),
        'range': (0., 1.),
        'weight': torch.ones_like(input_tensor).numpy(),
        'density': True,
        'out': out_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["torch.histogram_2"] = get_torch_histogram_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.histogram_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histogram_2'.")

check_valid('torch.histogram', generated_inputs['torch.histogram_2'], lib="torch", suffix=2)
