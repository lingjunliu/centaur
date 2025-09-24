
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def pack_padded_sequence_inputs():
    list_of_inputs = []

    # Input 1: Basic case, batch_first=False, sorted lengths
    input_tensor = np.random.randn(5, 3, 2).astype(np.float32)
    lengths = np.array([5, 3, 1], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': False,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case, batch_first=True, sorted lengths
    input_tensor = np.random.randn(4, 6, 5).astype(np.float32)
    lengths = np.array([6, 4, 4, 2], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': True,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unsorted lengths, batch_first=False, enforce_sorted=False
    input_tensor = np.random.randn(7, 4, 3).astype(np.float32)
    lengths = np.array([3, 7, 2, 5], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': False,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Unsorted lengths, batch_first=True, enforce_sorted=False
    input_tensor = np.random.randn(5, 8, 1).astype(np.float32)
    lengths = np.array([4, 8, 1, 3, 5], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': True,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All sequences have the same length, batch_first=True
    input_tensor = np.random.randn(3, 4, 4).astype(np.float32)
    lengths = np.array([4, 4, 4], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': True,
        'enforce_sorted': True # or False, doesn't matter
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: All sequences have the same length, batch_first=False
    input_tensor = np.random.randn(5, 2, 10).astype(np.float32)
    lengths = np.array([5, 5], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': False,
        'enforce_sorted': False # or True, doesn't matter
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single item in batch, batch_first=True
    input_tensor = np.random.randn(1, 5, 8).astype(np.float32)
    lengths = np.array([5], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': True,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single item in batch, batch_first=False
    input_tensor = np.random.randn(4, 1, 12).astype(np.float32)
    lengths = np.array([3], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': False,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64 input tensor, batch_first=True, unsorted lengths
    input_tensor = np.random.randn(6, 10, 2).astype(np.float64)
    lengths = np.array([9, 3, 10, 5, 2, 7], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': True,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sequences of length 1, batch_first=False, unsorted
    input_tensor = np.random.randn(4, 5, 1).astype(np.float32)
    lengths = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': False,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 2D input (no feature dimension), batch_first=True
    input_tensor = np.random.randn(3, 5).astype(np.float32)
    lengths = np.array([5, 2, 1], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': True,
        'enforce_sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D input (no feature dimension), batch_first=False
    input_tensor = np.random.randn(7, 4).astype(np.float32)
    lengths = np.array([3, 7, 2, 5], dtype=np.int64)
    input_dict = {
        'input': input_tensor,
        'lengths': lengths,
        'batch_first': False,
        'enforce_sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.utils.rnn.pack_padded_sequence"] = pack_padded_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.rnn.pack_padded_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pack_padded_sequence'.")

check_valid('torch.nn.utils.rnn.pack_padded_sequence', generated_inputs['torch.nn.utils.rnn.pack_padded_sequence'], lib="torch", suffix=0)
