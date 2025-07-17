
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def pack_padded_sequence_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[1, 2], [3, 0], [0, 0]], [[4, 5], [0, 0], [0, 0]], [[6, 7], [8, 9], [10, 11]]], dtype=np.int32)
    lengths = np.array([3, 1, 2], dtype=np.int64)
    batch_first = False
    enforce_sorted = False
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    lengths = np.array([3, 3], dtype=np.int64)
    batch_first = False
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], dtype=np.float64)
    lengths = np.array([2, 2, 2], dtype=np.int64)
    batch_first = True
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    lengths = np.array([3, 3], dtype=np.int64)
    batch_first = True
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2], [3, 0], [0, 0]], [[4, 5], [0, 0], [0, 0]], [[6, 7], [8, 9], [10, 11]]], dtype=np.int32)
    lengths = np.array([3, 1, 2], dtype=np.int64)
    batch_first = True
    enforce_sorted = False
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: all sequences have length 1
    input_tensor = np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.int32)
    lengths = np.array([1, 1, 1], dtype=np.int64)
    batch_first = False
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: batch_first = True
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [0, 0]]], dtype=np.int32)
    lengths = np.array([2, 2, 1], dtype=np.int64)
    batch_first = True
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: enforce_sorted = False, batch_first = False
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [0, 0]]], dtype=np.int32)
    lengths = np.array([2, 2, 1], dtype=np.int64)
    batch_first = False
    enforce_sorted = False
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Different shape
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [0, 0, 0]], [[16, 17, 18], [0,0,0],[0,0,0]]], dtype=np.float32)
    lengths = np.array([3, 2, 1], dtype=np.int64)
    batch_first = False
    enforce_sorted = False
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single batch
    input_tensor = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.int32)
    lengths = np.array([3], dtype=np.int64)
    batch_first = False
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: More varied example with enforce_sorted=True
    input_tensor = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [0, 0], [0, 0]], [[13, 14], [0, 0], [0, 0], [0, 0]]], dtype=np.int32)
    lengths = np.array([4, 2, 1], dtype=np.int64)
    batch_first = True
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: enforce_sorted=True and batch_first=False
    input_tensor = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [0, 0]], [[11, 12], [0, 0], [0, 0]]], dtype=np.int32)
    lengths = np.array([3, 2, 1], dtype=np.int64)
    batch_first = False
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Corrected the mismatch in batch size
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10],[11,12]]], dtype=np.int32)
    lengths = np.array([2, 2, 2], dtype=np.int64)
    batch_first = False
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: Corrected the mismatch in batch size and batch_first
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    lengths = np.array([2, 2], dtype=np.int64)
    batch_first = True
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15: Valid input with batch_first=True and sorted lengths
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]], dtype=np.int32)
    lengths = np.array([3, 3], dtype=np.int64)
    batch_first = True
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 16:
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 0, 0]]], dtype=np.int32)
    lengths = np.array([2, 1], dtype=np.int64)
    batch_first = False
    enforce_sorted = True
    input_dict = {"input": input_tensor, "lengths": lengths, "batch_first": batch_first, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.rnn.pack_padded_sequence"] = pack_padded_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.rnn.pack_padded_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pack_padded_sequence'.")

check_valid('torch.nn.utils.rnn.pack_padded_sequence', generated_inputs['torch.nn.utils.rnn.pack_padded_sequence'], lib="torch", suffix=0)
