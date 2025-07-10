
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_make_ndarray_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D tensor
    tensor1 = tf.make_tensor_proto(tf.constant([[1, 2, 3], [4, 5, 6]]))
    input_dict1 = {"tensor": tensor1}
    list_of_inputs.append(input_dict1)

    # Input 2: 1D tensor
    tensor2 = tf.make_tensor_proto(tf.constant([1, 2, 3, 4, 5]))
    input_dict2 = {"tensor": tensor2}
    list_of_inputs.append(input_dict2)

    # Input 3: 3D tensor
    tensor3 = tf.make_tensor_proto(tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    input_dict3 = {"tensor": tensor3}
    list_of_inputs.append(input_dict3)

    # Input 4: Tensor with float32
    tensor4 = tf.make_tensor_proto(tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32))
    input_dict4 = {"tensor": tensor4}
    list_of_inputs.append(input_dict4)

    # Input 5: Tensor with int64
    tensor5 = tf.make_tensor_proto(tf.constant([[1, 2], [3, 4]], dtype=tf.int64))
    input_dict5 = {"tensor": tensor5}
    list_of_inputs.append(input_dict5)

    # Input 6: Tensor with bool
    tensor6 = tf.make_tensor_proto(tf.constant([[True, False], [False, True]]))
    input_dict6 = {"tensor": tensor6}
    list_of_inputs.append(input_dict6)

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
