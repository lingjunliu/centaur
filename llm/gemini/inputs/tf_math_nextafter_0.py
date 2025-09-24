
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_nextafter_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensors
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 tensors
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2, "name": "test_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x2 = np.array([-4.0, -5.0, -6.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero values
    x1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values
    x1 = np.array([1e5, 2e5, 3e5], dtype=np.float32)
    x2 = np.array([4e5, 5e5, 6e5], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small values
    x1 = np.array([1e-5, 2e-5, 3e-5], dtype=np.float32)
    x2 = np.array([4e-5, 5e-5, 6e-5], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative values
    x1 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    x2 = np.array([4.0, -5.0, 6.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional float64
    x1 = np.random.rand(2, 3, 4).astype(np.float64)
    x2 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Same values
    x1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    x2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.nextafter"] = tf_math_nextafter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.nextafter'.")

check_valid('tf.math.nextafter', generated_inputs['tf.math.nextafter'], lib="tf", suffix=0)
