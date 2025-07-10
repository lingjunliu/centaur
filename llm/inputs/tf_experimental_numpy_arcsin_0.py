
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_arcsin_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = tf.constant(np.array([0.0, 0.5, -0.5]).astype(np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Input with a different data type
    x = tf.constant(np.array([0.2, -0.3]).astype(np.float64))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Input with a different shape (2D array)
    x = tf.constant(np.array([[0.1, 0.2], [-0.3, 0.4]]).astype(np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Input with a larger array
    x = tf.constant(np.linspace(-0.9, 0.9, 5).astype(np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Input with all positive values
    x = tf.constant(np.array([0.1, 0.3, 0.5]).astype(np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar input
    x = tf.constant(np.float32(0.6))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Input as tf.float16
    x = tf.constant(np.array([0.0, 0.25, 0.5]).astype(np.float16))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Input with values close to the boundaries (-1 and 1), excluding exactly -1 and 1
    x = tf.constant(np.array([-0.99, -0.9, 0.9, 0.99]).astype(np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor
    x = tf.constant(np.random.uniform(low=-0.9, high=0.9, size=(2, 3, 2)).astype(np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Input with negative values.
    x = tf.constant(np.array([-0.1, -0.3, -0.5]).astype(np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.arcsin"] = tf_experimental_numpy_arcsin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.arcsin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.arcsin'.")

check_valid('tf.experimental.numpy.arcsin', generated_inputs['tf.experimental.numpy.arcsin'], lib="tf", suffix=0)
