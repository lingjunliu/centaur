
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_arctan2_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive values
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Case with negative values
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x2 = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Case with zero values
    x1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x2 = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Case with different data types
    x1 = np.array([1, 2, 3], dtype=np.int32).astype(np.float32)
    x2 = np.array([1, 1, 1], dtype=np.int32).astype(np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Case with 2D arrays
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Case with 3D arrays
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    x2 = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Case with broadcasting
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array(1.0, dtype=np.float32)
    x2 = np.broadcast_to(x2, x1.shape)

    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative in 2D
    x1 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    x2 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger values
    x1 = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    x2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x2 close to zero
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([0.001, -0.001, 0.0], dtype=np.float32)
    input_dict = {"x1": tf.constant(x1), "x2": tf.constant(x2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.arctan2"] = tf_experimental_numpy_arctan2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.arctan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.arctan2'.")

check_valid('tf.experimental.numpy.arctan2', generated_inputs['tf.experimental.numpy.arctan2'], lib="tf", suffix=0)
