
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np


def nll_loss_inputs():
    list_of_inputs = []

    # Helper to generate realistic log-probability numpy arrays
    def get_log_probs(shape):
        return torch.nn.functional.log_softmax(torch.randn(shape), dim=1).numpy().astype(np.float32)

    # The user is facing a recurring cycle of errors:
    # 1. Omitting 'log_target' from the input dictionary causes a "KeyError: 'log_target'".
    # 2. Including 'log_target' causes a "TypeError: nll_loss() got multiple values for argument 'weight'".
    # This indicates the provided signature is faulty and conflicts with the actual PyTorch function signature,
    # with 'log_target' being incorrectly mapped to the positional 'weight' argument.
    # To resolve the immediate "KeyError: 'log_target'" as requested, this key must be included in all dictionaries.
    # This directly addresses the last reported error, even though it may cause the TypeError to reappear
    # due to the faulty test environment.
    
    # Input 1: Basic case with reduction='mean'
    input_dict_1 = {
        'input': get_log_probs((5, 4)),
        'target': np.random.randint(0, 4, (5,)).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32), 
        'weight': np.ones(4, dtype=np.float32),
        'size_average': False,
        'ignore_index': -100,
        'reduce': True,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with reduction='sum'
    input_dict_2 = {
        'input': get_log_probs((3, 10)),
        'target': np.random.randint(0, 10, (3,)).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.ones(10, dtype=np.float32),
        'size_average': False,
        'ignore_index': -100,
        'reduce': True,
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic case with reduction='none'
    input_dict_3 = {
        'input': get_log_probs((6, 2)),
        'target': np.array([0, 1, 0, 1, 0, 1]).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.ones(2, dtype=np.float32),
        'size_average': False,
        'ignore_index': -100,
        'reduce': True,
        'reduction': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With a custom weight tensor for class balancing
    input_dict_4 = {
        'input': get_log_probs((5, 3)),
        'target': np.random.randint(0, 3, (5,)).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.random.rand(3).astype(np.float32),
        'size_average': False,
        'ignore_index': -100,
        'reduce': True,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With an ignore_index to skip certain targets
    input_dict_5 = {
        'input': get_log_probs((8, 5)),
        'target': np.array([0, 1, 2, 3, 4, 2, 99, 1]).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.ones(5, dtype=np.float32),
        'size_average': False,
        'ignore_index': 99,
        'reduce': True,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: For 4D inputs (e.g., image segmentation)
    input_dict_6 = {
        'input': get_log_probs((2, 5, 4, 4)),
        'target': np.random.randint(0, 5, (2, 4, 4)).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.ones(5, dtype=np.float32),
        'size_average': False,
        'ignore_index': -100,
        'reduce': True,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D inputs with custom weight and ignore_index
    target_tensor_7 = np.random.randint(0, 6, (2, 10)).astype(np.int64)
    target_tensor_7[0, 3:6] = 99
    input_dict_7 = {
        'input': get_log_probs((2, 6, 10)),
        'target': target_tensor_7,
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.0]).astype(np.float32),
        'size_average': False,
        'ignore_index': 99,
        'reduce': True,
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using deprecated `size_average=True`
    input_dict_8 = {
        'input': get_log_probs((10, 3)),
        'target': np.random.randint(0, 3, (10,)).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.ones(3, dtype=np.float32),
        'size_average': True,
        'ignore_index': -100,
        'reduce': True,
        'reduction': 'mean' 
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using deprecated `reduce=False`
    input_dict_9 = {
        'input': get_log_probs((7, 4)),
        'target': np.random.randint(0, 4, (7,)).astype(np.int64),
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.ones(4, dtype=np.float32),
        'size_average': False,
        'ignore_index': -100,
        'reduce': False,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 4D input with weight and a different ignore_index
    target_tensor_10 = np.random.randint(1, 4, (2, 3, 3)).astype(np.int64)
    target_tensor_10[0, 1, 1] = 0 
    input_dict_10 = {
        'input': get_log_probs((2, 4, 3, 3)),
        'target': target_tensor_10,
        'log_target': np.array([0], dtype=np.float32),
        'weight': np.random.rand(4).astype(np.float32),
        'size_average': False,
        'ignore_index': 0,
        'reduce': True,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.nn.functional.nll_loss_2"] = nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.nll_loss_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.nll_loss_2'.")

check_valid('torch.nn.functional.nll_loss', generated_inputs['torch.nn.functional.nll_loss_2'], lib="torch", suffix=2)
