
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def get_full_embedding_bag_dict():
    """Returns a dictionary with all keys from the signature and safe defaults."""
    input_tensor = np.array([0], dtype=np.int64)
    return {
        'input': input_tensor,
        'weight': np.zeros((1, 1), dtype=np.float32),
        'indices': np.copy(input_tensor),
        'offsets': np.array([0], dtype=np.int64),
        'max_norm': 1e12, # A non-interfering default float
        'norm_type': 2.0,
        'scale_grad_by_freq': False,
        'mode': 'mean',
        'sparse': False,
        'per_sample_weights': np.array([], dtype=np.float32), # Empty tensor treated as None
        'include_last_offset': False,
    }

def embedding_bag_inputs():
    list_of_inputs = []

    # Input 1: Basic case with mode='sum'
    input_dict_1 = get_full_embedding_bag_dict()
    input_tensor_1 = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    input_dict_1.update({
        'input': input_tensor_1,
        'weight': np.random.randn(10, 3).astype(np.float32),
        'offsets': np.array([0, 4], dtype=np.int64),
        'indices': np.copy(input_tensor_1),
        'mode': 'sum',
    })
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with mode='mean'
    input_dict_2 = get_full_embedding_bag_dict()
    input_tensor_2 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    input_dict_2.update({
        'input': input_tensor_2,
        'weight': np.random.randn(6, 8).astype(np.float32),
        'offsets': np.array([0, 2, 5], dtype=np.int64),
        'indices': np.copy(input_tensor_2),
        'mode': 'mean',
    })
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic case with mode='max'
    input_dict_3 = get_full_embedding_bag_dict()
    input_tensor_3 = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    input_dict_3.update({
        'input': input_tensor_3,
        'weight': np.random.randn(10, 3).astype(np.float32),
        'offsets': np.array([0, 4], dtype=np.int64),
        'indices': np.copy(input_tensor_3),
        'mode': 'max',
    })
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: include_last_offset=True
    input_dict_4 = get_full_embedding_bag_dict()
    input_tensor_4 = np.array([0, 1, 2, 3], dtype=np.int64)
    input_dict_4.update({
        'input': input_tensor_4,
        'weight': np.random.randn(5, 5).astype(np.float32),
        'offsets': np.array([0, 2, 4], dtype=np.int64),
        'indices': np.copy(input_tensor_4),
        'include_last_offset': True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With per_sample_weights (requires mode='sum')
    input_dict_5 = get_full_embedding_bag_dict()
    input_tensor_5 = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    per_sample_weights_5 = np.random.rand(8).astype(np.float32)
    input_dict_5.update({
        'input': input_tensor_5,
        'weight': np.random.randn(10, 3).astype(np.float32),
        'offsets': np.array([0, 4], dtype=np.int64),
        'indices': np.copy(input_tensor_5),
        'per_sample_weights': per_sample_weights_5,
        'mode': 'sum',
    })
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With max_norm and norm_type
    input_dict_6 = get_full_embedding_bag_dict()
    input_tensor_6 = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    input_dict_6.update({
        'input': input_tensor_6,
        'weight': (np.random.randn(10, 3) * 5).astype(np.float32),
        'offsets': np.array([0, 4], dtype=np.int64),
        'indices': np.copy(input_tensor_6),
        'max_norm': 1.0,
        'norm_type': 2.0,
    })
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: scale_grad_by_freq=True
    input_dict_7 = get_full_embedding_bag_dict()
    input_tensor_7 = np.array([1, 2, 4, 5, 4, 3, 2, 9, 2], dtype=np.int64)
    input_dict_7.update({
        'input': input_tensor_7,
        'weight': np.random.randn(10, 3).astype(np.float32),
        'offsets': np.array([0, 4, 9], dtype=np.int64),
        'indices': np.copy(input_tensor_7),
        'scale_grad_by_freq': True,
        'include_last_offset': True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: sparse=True
    input_dict_8 = get_full_embedding_bag_dict()
    input_tensor_8 = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    input_dict_8.update({
        'input': input_tensor_8,
        'weight': np.random.randn(10, 3).astype(np.float32),
        'offsets': np.array([0, 4], dtype=np.int64),
        'indices': np.copy(input_tensor_8),
        'sparse': True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: With an empty bag
    input_dict_9 = get_full_embedding_bag_dict()
    input_tensor_9 = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict_9.update({
        'input': input_tensor_9,
        'weight': np.random.randn(5, 2).astype(np.float32),
        'offsets': np.array([0, 0, 2, 4], dtype=np.int64),
        'indices': np.copy(input_tensor_9),
        'mode': 'mean',
        'include_last_offset': True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Empty input tensor
    input_dict_10 = get_full_embedding_bag_dict()
    input_tensor_10 = np.array([], dtype=np.int64)
    offsets_tensor_10 = np.array([0, 0, 0], dtype=np.int64)
    input_dict_10.update({
        'input': input_tensor_10,
        'weight': np.random.randn(10, 4).astype(np.float32),
        'offsets': offsets_tensor_10,
        'indices': np.copy(input_tensor_10),
        'include_last_offset': True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: All arguments combined (Corrected: mode='sum' because per_sample_weights is used)
    input_dict_11 = get_full_embedding_bag_dict()
    input_tensor_11 = np.arange(15, dtype=np.int64)
    per_sample_weights_11 = np.random.rand(15).astype(np.float32)
    input_dict_11.update({
        'input': input_tensor_11,
        'weight': np.random.randn(20, 10).astype(np.float32),
        'offsets': np.array([0, 5, 10, 15], dtype=np.int64),
        'indices': np.copy(input_tensor_11),
        'per_sample_weights': per_sample_weights_11,
        'mode': 'sum', # Corrected from 'mean'
        'max_norm': 2.5,
        'norm_type': 1.5,
        'scale_grad_by_freq': True,
        'sparse': True,
        'include_last_offset': True,
    })
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding_bag_4"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.embedding_bag_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding_bag_4'.")

check_valid('torch.nn.functional.embedding_bag', generated_inputs['torch.nn.functional.embedding_bag_4'], lib="torch", suffix=4)
