
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_bucket_by_sequence_length_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [5, 10, 15],
        "bucket_batch_sizes": [2, 3, 4, 5],
        "padded_shapes": (tf.TensorShape([None]),),
        "padding_values": np.int64(0),
        "pad_to_bucket_boundary": False,
        "no_padding": False,
        "drop_remainder": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [3, 7],
        "bucket_batch_sizes": [4, 2, 3],
        "padded_shapes": (tf.TensorShape([None]),),
        "padding_values": np.int32(-1),
        "pad_to_bucket_boundary": True,
        "no_padding": False,
        "drop_remainder": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [2, 4, 6, 8],
        "bucket_batch_sizes": [1, 2, 3, 4, 5],
        "padded_shapes": (tf.TensorShape([None]),),
        "padding_values": np.float32(100),
        "pad_to_bucket_boundary": False,
        "no_padding": False,
        "drop_remainder": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [10],
        "bucket_batch_sizes": [5, 6],
        "padded_shapes": (tf.TensorShape([None, None]),),
        "padding_values": np.int32(0),
        "pad_to_bucket_boundary": True,
        "no_padding": False,
        "drop_remainder": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [5, 10],
        "bucket_batch_sizes": [2, 3, 4],
        "padded_shapes": (tf.TensorShape([None]), tf.TensorShape([None])),
        "padding_values": np.array([0, 0], dtype=np.int64),
        "pad_to_bucket_boundary": False,
        "no_padding": False,
        "drop_remainder": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [1, 2, 3],
        "bucket_batch_sizes": [4, 5, 6, 7],
        "padded_shapes": (tf.TensorShape([None]),),
        "padding_values": np.int32(0),
        "pad_to_bucket_boundary": True,
        "no_padding": False,
        "drop_remainder": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [4],
        "bucket_batch_sizes": [3, 2],
        "padded_shapes": (tf.TensorShape([None]),),
        "padding_values": np.float64(-1),
        "pad_to_bucket_boundary": False,
        "no_padding": False,
        "drop_remainder": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [6, 12],
        "bucket_batch_sizes": [1, 2, 3],
        "padded_shapes": (tf.TensorShape([None, None, None]),),
        "padding_values": np.int32(0),
        "pad_to_bucket_boundary": True,
        "no_padding": False,
        "drop_remainder": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [7, 14, 21],
        "bucket_batch_sizes": [4, 5, 6, 7],
        "padded_shapes": (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),
        "padding_values": np.array([0, 0, 0], dtype=np.int64),
        "pad_to_bucket_boundary": False,
        "no_padding": False,
        "drop_remainder": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [8, 16],
        "bucket_batch_sizes": [1, 2, 3],
        "padded_shapes": (tf.TensorShape([None]),),
        "padding_values": np.int32(1),
        "pad_to_bucket_boundary": True,
        "no_padding": False,
        "drop_remainder": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "element_length_func": [lambda elem: elem.shape[0]],
        "bucket_boundaries": [20, 30, 40],
        "bucket_batch_sizes": [2, 3, 4, 5],
        "padded_shapes": (tf.TensorShape([None, None]),),
        "padding_values": np.int64(-1),
        "pad_to_bucket_boundary": False,
        "no_padding": False,
        "drop_remainder": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.bucket_by_sequence_length"] = tf_data_experimental_bucket_by_sequence_length_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.bucket_by_sequence_length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.bucket_by_sequence_length'.")

check_valid('tf.data.experimental.bucket_by_sequence_length', generated_inputs['tf.data.experimental.bucket_by_sequence_length'], lib="tf", suffix=0)
