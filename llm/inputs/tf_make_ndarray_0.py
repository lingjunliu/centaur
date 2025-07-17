
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_make_ndarray_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    a = tf.constant([1, 2, 3, 4, 5])
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with different data type
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: array of booleans
    a = tf.constant([True, False, True])
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: array of int64
    a = tf.constant([1, 2, 3], dtype=tf.int64)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: array of float64
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Negative values
    a = tf.constant([-1, -2, -3], dtype=tf.int32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 2D negative values
    a = tf.constant([[-1, 2], [-3, 4]], dtype=tf.int32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.make_ndarray"] = tf_make_ndarray_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.make_ndarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.make_ndarray'.")

check_valid('tf.make_ndarray', generated_inputs['tf.make_ndarray'], lib="tf", suffix=0)
