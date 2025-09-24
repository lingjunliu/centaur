
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_dot_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    a = tf.constant(np.array([1, 2, 3]))
    b = tf.constant(np.array([4, 5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D arrays
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    b = tf.constant(np.array([[5, 6], [7, 8]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix and vector
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    b = tf.constant(np.array([5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Vector and matrix
    a = tf.constant(np.array([1, 2]))
    b = tf.constant(np.array([[3, 4], [5, 6]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes for 2D matrices, valid for dot product
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    b = tf.constant(np.array([[7, 8], [9, 10], [11, 12]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    a = tf.constant(np.array([[-1, 2], [3, -4]]))
    b = tf.constant(np.array([[5, -6], [-7, 8]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large values
    a = tf.constant(np.array([[1000, 2000], [3000, 4000]]))
    b = tf.constant(np.array([[5000, 6000], [7000, 8000]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero values
    a = tf.constant(np.array([[0, 2], [3, 0]]))
    b = tf.constant(np.array([[0, 6], [7, 0]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D Tensors that can be dotted
    a = tf.constant(np.random.rand(2,3,4))
    b = tf.constant(np.random.rand(2,4,5))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D Tensors that can be dotted
    a = tf.constant(np.random.rand(2,3,4,5))
    b = tf.constant(np.random.rand(2,3,5,6))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.dot"] = tf_experimental_numpy_dot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.dot'.")

check_valid('tf.experimental.numpy.dot', generated_inputs['tf.experimental.numpy.dot'], lib="tf", suffix=0)
