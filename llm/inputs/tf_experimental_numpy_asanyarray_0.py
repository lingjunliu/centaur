
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_asanyarray_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic int32 tensor
    a = tf.constant([1, 2, 3], dtype=tf.int32).numpy()
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32).numpy()
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float64).numpy()
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values
    a = tf.constant([-1, -2, 3], dtype=tf.int32).numpy()
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with boolean values
    a = tf.constant([True, False, True], dtype=tf.bool).numpy()
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with mixed positive and negative values
    a = tf.constant([-1.0, 2.0, -3.0], dtype=tf.float32).numpy()
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor
    a = tf.constant([], dtype=tf.int32).numpy()
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 4 tensor
    a = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=tf.int32).numpy()
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: tf.string tensor
    a = tf.constant(["a", "b", "c"], dtype=tf.string).numpy()
    dtype = np.object_
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.asanyarray"] = tf_experimental_numpy_asanyarray_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.asanyarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.asanyarray'.")

check_valid('tf.experimental.numpy.asanyarray', generated_inputs['tf.experimental.numpy.asanyarray'], lib="tf", suffix=0)
