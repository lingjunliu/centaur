
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_shape_n_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensors
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    b = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D tensors
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
    b = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32))
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int64, "name": "shape_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D Tensors
    a = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    b = tf.constant(np.array([6, 7, 8, 9, 10], dtype=np.int32))
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensors with different shapes
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    b = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single tensor in the list
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
    input_list = [a]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensors
    a = tf.constant(np.random.rand(2, 3, 4, 5).astype(np.float32))
    b = tf.constant(np.random.rand(2, 3, 4, 5).astype(np.float32))

    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalars
    a = tf.constant(5)
    b = tf.constant(10.0)
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data types

    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    b = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int64, "name": "shape_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values
    a = tf.constant(np.array([[-1, 2], [3, -4]], dtype=np.int32))
    b = tf.constant(np.array([[5, -6], [-7, 8]], dtype=np.int32))
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 0
    a = tf.constant(np.array(10, dtype=np.int32))
    b = tf.constant(np.array(5.0, dtype=np.float32))
    input_list = [a, b]
    input_dict = {"input": input_list, "out_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.shape_n"] = tf_shape_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.shape_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.shape_n'.")

check_valid('tf.shape_n', generated_inputs['tf.shape_n'], lib="tf", suffix=0)
