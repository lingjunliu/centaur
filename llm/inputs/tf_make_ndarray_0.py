
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_make_ndarray_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    tensor_proto = tf.make_tensor_proto(a)
    input_dict = {"tensor": tensor_proto}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    tensor_proto = tf.make_tensor_proto(a)
    input_dict = {"tensor": tensor_proto}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D complex64 array
    a = tf.constant([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=tf.complex64)
    tensor_proto = tf.make_tensor_proto(a)
    input_dict = {"tensor": tensor_proto}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.make_ndarray"] = tf_make_ndarray_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.make_ndarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.make_ndarray'.")

check_valid('tf.make_ndarray', generated_inputs['tf.make_ndarray'], lib="tf", suffix=0)
