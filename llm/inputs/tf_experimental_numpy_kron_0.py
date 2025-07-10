
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_kron_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    b = tf.constant(np.array([[5, 6], [7, 8]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes
    a = tf.constant(np.array([1, 2, 3]))
    b = tf.constant(np.array([[4, 5], [6, 7]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalars
    a = tf.constant(np.array(2))
    b = tf.constant(np.array(3))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimensions
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    b = tf.constant(np.array([[9, 10], [11, 12]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different dtypes (int32)
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    b = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtypes (float32)
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    b = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.float32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    a = tf.constant(np.array([[-1, 2], [3, -4]]))
    b = tf.constant(np.array([[5, -6], [-7, 8]]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One of the inputs is a vector of size one
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    b = tf.constant(np.array([5]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 1 arrays
    a = tf.constant(np.array([1, 2, 3]))
    b = tf.constant(np.array([4, 5]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty arrays - avoiding this as numpy min/max doesn't work well with it for tf tensors
    # a = tf.constant(np.array([]))
    # b = tf.constant(np.array([]))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.kron"] = tf_experimental_numpy_kron_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.kron'.")

check_valid('tf.experimental.numpy.kron', generated_inputs['tf.experimental.numpy.kron'], lib="tf", suffix=0)
