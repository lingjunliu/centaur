
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_make_ndarray_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with float32
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with negative values
    a = tf.constant([-1, -2, -3], dtype=tf.int32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with boolean values
    a = tf.constant([True, False, True], dtype=tf.bool)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with string values
    a = tf.constant(["hello", "world"], dtype=tf.string)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with int64 values
    a = tf.constant([123456789012345, 987654321098765], dtype=tf.int64)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with float64 values
    a = tf.constant([1.123456789, 2.987654321], dtype=tf.float64)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: scalar tensor
    a = tf.constant(5, dtype=tf.int32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 9: 3D Tensor
    a = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=tf.int32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 10: 1x1 Tensor
    a = tf.constant([[5]], dtype=tf.int32)
    proto_tensor = tf.make_tensor_proto(a)
    input_dict = {"tensor": proto_tensor}
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
