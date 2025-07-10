
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D matrices
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    x2 = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix and vector
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    x2 = tf.constant(np.array([5, 6], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector and matrix
    x1 = tf.constant(np.array([1, 2], dtype=np.float64))
    x2 = tf.constant(np.array([[3, 4], [5, 6]], dtype=np.float64))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched matrices
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    x2 = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int64))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D and 2D
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
    x2 = tf.constant(np.array([[9, 10], [11, 12]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D and 3D
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float64))
    x2 = tf.constant(np.array([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float64))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Negative values
    x1 = tf.constant(np.array([[-1, 2], [3, -4]], dtype=np.int32))
    x2 = tf.constant(np.array([[5, -6], [-7, 8]], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  1D vectors
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    x2 = tf.constant(np.array([4, 5, 6], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shapes that are incompatible
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    x2 = tf.constant(np.array([[5, 6], [7, 8], [9, 10]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different dtypes
    x1 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    x2 = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.matmul"] = tf_experimental_numpy_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.matmul'.")

check_valid('tf.experimental.numpy.matmul', generated_inputs['tf.experimental.numpy.matmul'], lib="tf", suffix=0)
