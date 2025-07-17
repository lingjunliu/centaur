
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def pack_sequence_inputs():
    list_of_inputs = []

    # Input 1: Basic case, sorted sequences
    seq1 = torch.tensor([1, 2, 3]).numpy()
    seq2 = torch.tensor([4, 5]).numpy()
    seq3 = torch.tensor([6]).numpy()
    sequences = [seq1, seq2, seq3]
    enforce_sorted = True
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Unsorted sequences, enforce_sorted=False
    seq1 = torch.tensor([1, 2]).numpy()
    seq2 = torch.tensor([3, 4, 5]).numpy()
    seq3 = torch.tensor([6]).numpy()
    sequences = [seq1, seq2, seq3]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sequences of different data types
    seq1 = torch.tensor([1.0, 2.0]).numpy()
    seq2 = torch.tensor([3.0, 4.0, 5.0]).numpy()
    sequences = [seq1, seq2]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sorted sequences with a single sequence
    seq1 = torch.tensor([1, 2, 3, 4]).numpy()
    sequences = [seq1]
    enforce_sorted = True
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sequences with negative values
    seq1 = torch.tensor([-1, 2, -3]).numpy()
    seq2 = torch.tensor([4, -5]).numpy()
    sequences = [seq1, seq2]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sequences of same length
    seq1 = torch.tensor([1, 2, 3]).numpy()
    seq2 = torch.tensor([4, 5, 6]).numpy()
    sequences = [seq1, seq2]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D sequences of different lengths
    seq1 = torch.tensor([1]).numpy()
    seq2 = torch.tensor([2,3]).numpy()
    seq3 = torch.tensor([4,5,6]).numpy()
    sequences = [seq1,seq2,seq3]
    enforce_sorted = True
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D sequences of different lengths, enforce_sorted=False
    seq1 = torch.tensor([2,3]).numpy()
    seq2 = torch.tensor([1]).numpy()
    seq3 = torch.tensor([4,5,6]).numpy()
    sequences = [seq1,seq2,seq3]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty sequences, enforce_sorted=False
    seq1 = torch.tensor([]).numpy()
    seq2 = torch.tensor([]).numpy()
    sequences = [seq1, seq2]
    enforce_sorted = False
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Empty sequences, enforce_sorted=True
    seq1 = torch.tensor([]).numpy()
    seq2 = torch.tensor([]).numpy()
    sequences = [seq1, seq2]
    enforce_sorted = True
    input_dict = {"sequences": sequences, "enforce_sorted": enforce_sorted}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.rnn.pack_sequence"] = pack_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.rnn.pack_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pack_sequence'.")

check_valid('torch.nn.utils.rnn.pack_sequence', generated_inputs['torch.nn.utils.rnn.pack_sequence'], lib="torch", suffix=0)
