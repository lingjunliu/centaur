
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_2d_inputs():
    list_of_inputs = []

    # Helper function to convert to numpy array if necessary
    def to_numpy(x):
        if isinstance(x, tf.Tensor):
            return x.numpy()
        return x

    # Input 1: Scalar
    arys = [tf.constant(5)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    arys = [tf.constant([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Already 2D
    arys = [tf.constant([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple arrays, scalar and 1D
    arys = [tf.constant(5), tf.constant([1, 2, 3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple arrays, 1D and 2D
    arys = [tf.constant([1, 2, 3]), tf.constant([[1, 2], [3, 4]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple arrays, all scalars
    arys = [tf.constant(1), tf.constant(2), tf.constant(3)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array
    arys = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed dtypes
    arys = [tf.constant(1, dtype=tf.int32), tf.constant([1.0, 2.0], dtype=tf.float32)]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values
    arys = [tf.constant([-1, -2, -3])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Bool type
    arys = [tf.constant([True, False])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty tensor with shape
    arys = [tf.constant([], dtype=tf.float32, shape=[0])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Multiple empty tensors
    arys = [tf.constant([], dtype=tf.float32, shape=[0]), tf.constant([], dtype=tf.int32, shape=[0])]
    input_dict = {"arys": arys}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.atleast_2d"] = tf_experimental_numpy_atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.atleast_2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_2d'.")

check_valid('tf.experimental.numpy.atleast_2d', generated_inputs['tf.experimental.numpy.atleast_2d'], lib="tf", suffix=0)
