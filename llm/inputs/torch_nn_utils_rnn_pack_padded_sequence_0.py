
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def pack_padded_sequence_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[[1, 2], [3, 0]],
                       [[4, 5], [6, 7]]], dtype=np.float32)
    lengths1 = np.array([2, 2], dtype=np.int64)
    batch_first1 = False
    enforce_sorted1 = True
    input_dict1 = {"input": input1, "lengths": lengths1, "batch_first": batch_first1, "enforce_sorted": enforce_sorted1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [0, 0]]], dtype=np.float32)
    lengths2 = np.array([2, 2, 1], dtype=np.int64)
    batch_first2 = False
    enforce_sorted2 = False
    input_dict2 = {"input": input2, "lengths": lengths2, "batch_first": batch_first2, "enforce_sorted": enforce_sorted2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[1, 2, 3], [4, 5, 6]], [[10, 11, 12], [13, 14, 15]]], dtype=np.float32)
    lengths3 = np.array([2, 2], dtype=np.int64)
    batch_first3 = True
    enforce_sorted3 = True
    input_dict3 = {"input": input3, "lengths": lengths3, "batch_first": batch_first3, "enforce_sorted": enforce_sorted3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    lengths4 = np.array([2, 2], dtype=np.int64)
    batch_first4 = False
    enforce_sorted4 = True
    input_dict4 = {"input": input4, "lengths": lengths4, "batch_first": batch_first4, "enforce_sorted": enforce_sorted4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.array([[[1, 2]], [[3, 4]]], dtype=np.float32)
    lengths5 = np.array([1, 1], dtype=np.int64)
    batch_first5 = False
    enforce_sorted5 = True
    input_dict5 = {"input": input5, "lengths": lengths5, "batch_first": batch_first5, "enforce_sorted": enforce_sorted5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[13, 14, 15, 16], [17, 18, 19, 20]]], dtype=np.float32)
    lengths6 = np.array([2, 2], dtype=np.int64)
    batch_first6 = True
    enforce_sorted6 = False
    input_dict6 = {"input": input6, "lengths": lengths6, "batch_first": batch_first6, "enforce_sorted": enforce_sorted6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    lengths7 = np.array([2, 2], dtype=np.int64)
    batch_first7 = True
    enforce_sorted7 = True
    input_dict7 = {"input": input7, "lengths": lengths7, "batch_first": batch_first7, "enforce_sorted": enforce_sorted7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.array([[[1.0, 2.0], [3.0, 0.0]], [[4.0, 5.0], [6.0, 0.0]]], dtype=np.float32)
    lengths8 = np.array([2, 1], dtype=np.int64)
    batch_first8 = False
    enforce_sorted8 = True
    input_dict8 = {"input": input8, "lengths": lengths8, "batch_first": batch_first8, "enforce_sorted": enforce_sorted8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9. Fixed issue with enforce_sorted, and removed the last row to avoid indexing issues
    input9 = np.array([[[1, 2, 3], [4, 5, 0]], [[6, 7, 8], [9, 0, 0]]], dtype=np.float32)
    lengths9 = np.array([2, 2], dtype=np.int64)
    batch_first9 = True
    enforce_sorted9 = True  # Corrected to True since we are enforcing sorted
    input_dict9 = {"input": input9, "lengths": lengths9, "batch_first": batch_first9, "enforce_sorted": enforce_sorted9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.array([[[1]], [[2]], [[3]]], dtype=np.float32)
    lengths10 = np.array([1, 1, 1], dtype=np.int64)
    batch_first10 = False
    enforce_sorted10 = True
    input_dict10 = {"input": input10, "lengths": lengths10, "batch_first": batch_first10, "enforce_sorted": enforce_sorted10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11 added to avoid only having smaller sequence lengths
    input11 = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [0,0]]], dtype=np.float32)
    lengths11 = np.array([3, 3, 2], dtype=np.int64)
    batch_first11 = True
    enforce_sorted11 = True
    input_dict11 = {"input": input11, "lengths": lengths11, "batch_first": batch_first11, "enforce_sorted": enforce_sorted11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12 added with enforce_sorted = False
    input12 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    lengths12 = np.array([2, 2], dtype=np.int64)
    batch_first12 = False
    enforce_sorted12 = False
    input_dict12 = {"input": input12, "lengths": lengths12, "batch_first": batch_first12, "enforce_sorted": enforce_sorted12}
    list_of_inputs.append(copy.deepcopy(input_dict12))

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
