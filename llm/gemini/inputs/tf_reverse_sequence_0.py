
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_sequence_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
    seq_lengths_tensor = np.array([3, 2], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_1"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    seq_lengths_tensor = np.array([1, 2], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_2"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    seq_lengths_tensor = np.array([2, 1, 3], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_3"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    seq_lengths_tensor = np.array([4, 1], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_4"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.int32)
    seq_lengths_tensor = np.array([6, 6], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_5"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    seq_lengths_tensor = np.array([2, 1], dtype=np.int64)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_6"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    seq_lengths_tensor = np.array([2, 1], dtype=np.int64)
    seq_axis_val = 2
    batch_axis_val = 0
    name_val = "reverse_seq_7"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    seq_lengths_tensor = np.array([4, 1], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_8"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
    seq_lengths_tensor = np.array([3, 2], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_9"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    seq_lengths_tensor = np.array([2, 3], dtype=np.int32)
    seq_axis_val = 1
    batch_axis_val = 0
    name_val = "reverse_seq_10"

    input_dict = {
        "input": input_tensor,
        "seq_lengths": seq_lengths_tensor,
        "seq_axis": seq_axis_val,
        "batch_axis": batch_axis_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.reverse_sequence"] = tf_reverse_sequence_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.reverse_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse_sequence'.")

check_valid('tf.reverse_sequence', generated_inputs['tf.reverse_sequence'], lib="tf", suffix=0)
