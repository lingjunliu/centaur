
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_log_inputs():
    list_of_inputs = []

    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Input 1: Basic positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive values close to zero
    x = np.array([1e-5, 1e-10, 1e-15], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with mixed values
    x = np.array([0.1, 1.0, 10.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with integer type, which will be implicitly converted to float
    x = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single value
    x = np.array(5.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([2.5, 7.2, 10.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 5, 10], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.log"] = tf_experimental_numpy_log_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.log'.")

check_valid('tf.experimental.numpy.log', generated_inputs['tf.experimental.numpy.log'], lib="tf", suffix=0)
