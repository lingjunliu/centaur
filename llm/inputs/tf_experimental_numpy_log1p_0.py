
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_log1p_inputs():
    list_of_inputs = []

    # Input 1: Scalar input
    x = np.array(0.0, dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive value
    x = np.array(1.0, dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative value (close to -1)
    x = np.array(-0.5, dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero
    x = np.array(0.0, dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive value
    x = np.array(100.0, dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array
    x = np.array([0.0, 1.0, 2.0, -0.5], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array
    x = np.array([[0.0, 1.0], [2.0, -0.5]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array
    x = np.array([[[0.0, 1.0], [2.0, -0.5]], [[0.5, 1.5], [2.5, -0.1]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Using float64
    x = np.array(0.0, dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using negative value close to -1, float64
    x = np.array(-0.99, dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.log1p"] = tf_experimental_numpy_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.log1p'.")

check_valid('tf.experimental.numpy.log1p', generated_inputs['tf.experimental.numpy.log1p'], lib="tf", suffix=0)
