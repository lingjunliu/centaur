
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_transpose_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array transpose
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    axes = (1, 0)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array transpose
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    axes = (0, 2, 1)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array transpose (should not raise error but effectively no change)
    a = tf.constant(np.array([1, 2, 3]))
    axes = (0,)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array transpose
    a = tf.constant(np.random.rand(2, 3, 4, 5).astype(np.float32))
    axes = (3, 1, 0, 2)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty array
    a = tf.constant(np.array([]))
    axes = ()
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Different data types
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int64))
    axes = (1, 0)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with different axes order
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    axes = (2, 0, 1)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Transpose with identical axes
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    axes = (0, 1)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: another 4D
    a = tf.constant(np.random.rand(5, 4, 3, 2).astype(np.float32))
    axes = (1, 3, 0, 2)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: test with dimension with value of 1
    a = tf.constant(np.random.rand(1, 4, 3, 2).astype(np.float32))
    axes = (1, 3, 0, 2)
    input_dict = {"a": a, "axes": tuple(np.int32(axes))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.transpose"] = tf_experimental_numpy_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.transpose'.")

check_valid('tf.experimental.numpy.transpose', generated_inputs['tf.experimental.numpy.transpose'], lib="tf", suffix=0)
