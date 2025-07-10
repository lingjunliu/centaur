
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_range_inputs():
    list_of_inputs = []

    # Input 1
    starts = np.array([3, 5, 2], dtype=np.int32)
    limits = None
    deltas = np.array(1, dtype=np.int32)
    dtype = tf.int32
    name = "range1"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int32).numpy(),
        "limits": tf.convert_to_tensor(np.array([0,0,0], dtype=np.int32)).numpy() if limits is None else tf.convert_to_tensor(limits, dtype=tf.int32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    starts = np.array([0, 5, 8], dtype=np.int32)
    limits = np.array([3, 3, 12], dtype=np.int32)
    deltas = np.array(2, dtype=np.int32)
    dtype = tf.int32
    name = "range2"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int32).numpy(),
        "limits": tf.convert_to_tensor(limits, dtype=tf.int32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    starts = np.array([0, -5, 8], dtype=np.int32)
    limits = np.array([3, -3, 12], dtype=np.int32)
    deltas = np.array(2, dtype=np.int32)
    dtype = tf.int32
    name = "range3"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int32).numpy(),
        "limits": tf.convert_to_tensor(limits, dtype=tf.int32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    starts = np.array([0, 5, 8], dtype=np.int32)
    limits = np.array([3, 3, 12], dtype=np.int32)
    deltas = np.array([1, 2, 3], dtype=np.int32)
    dtype = tf.int32
    name = "range4"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int32).numpy(),
        "limits": tf.convert_to_tensor(limits, dtype=tf.int32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    starts = np.array(5, dtype=np.int32)
    limits = np.array([3, 3, 12], dtype=np.int32)
    deltas = np.array([1, 2, 3], dtype=np.int32)
    dtype = tf.int32
    name = "range5"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int32).numpy(),
        "limits": tf.convert_to_tensor(limits, dtype=tf.int32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    starts = np.array([3, 5, 2], dtype=np.int64)
    limits = None
    deltas = np.array(1, dtype=np.int64)
    dtype = tf.int64
    name = "range6"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int64).numpy(),
        "limits": tf.convert_to_tensor(np.array([0,0,0], dtype=np.int64)).numpy() if limits is None else tf.convert_to_tensor(limits, dtype=tf.int64).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int64).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    starts = np.array([0, 5, 8], dtype=np.float32)
    limits = np.array([3, 3, 12], dtype=np.float32)
    deltas = np.array(2, dtype=np.float32)
    dtype = tf.float32
    name = "range7"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.float32).numpy(),
        "limits": tf.convert_to_tensor(limits, dtype=tf.float32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.float32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    starts = np.array([0.0, 5.0, 8.0], dtype=np.float64)
    limits = np.array([3.0, 3.0, 12.0], dtype=np.float64)
    deltas = np.array(2.0, dtype=np.float64)
    dtype = tf.float64
    name = "range8"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.float64).numpy(),
        "limits": tf.convert_to_tensor(limits, dtype=tf.float64).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.float64).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    starts = np.array([3, 5, 2], dtype=np.int32)
    limits = None
    deltas = np.array(1, dtype=np.int32)
    dtype = tf.int32
    name = "range9"
    row_splits_dtype = tf.int32

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int32).numpy(),
        "limits": tf.convert_to_tensor(np.array([0,0,0], dtype=np.int32)).numpy() if limits is None else tf.convert_to_tensor(limits, dtype=tf.int32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    starts = np.array([0, 5, 8], dtype=np.int32)
    limits = np.array([3, 3, 12], dtype=np.int32)
    deltas = np.array([1, -2, 3], dtype=np.int32)
    dtype = tf.int32
    name = "range10"
    row_splits_dtype = tf.int64

    input_dict = {
        "starts": tf.convert_to_tensor(starts, dtype=tf.int32).numpy(),
        "limits": tf.convert_to_tensor(limits, dtype=tf.int32).numpy(),
        "deltas": tf.convert_to_tensor(deltas, dtype=tf.int32).numpy(),
        "dtype": dtype,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.range"] = tf_ragged_range_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.range' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.range'.")

check_valid('tf.ragged.range', generated_inputs['tf.ragged.range'], lib="tf", suffix=0)
