
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_digamma_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(1.0)
    name = "digamma_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, vector
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "digamma_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, matrix
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    name = "digamma_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, tensor
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    name = "digamma_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "digamma_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, large values
    x = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
    name = "digamma_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    name = "digamma_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, mixed positive and negative
    x = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float16)
    name = "digamma_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, multi-dimensional array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    name = "digamma_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, a single large value in a multi-dimensional array
    x = np.array([[[1.0, 2.0], [3.0, 1e10]]], dtype=np.float64)
    name = "digamma_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Digamma"] = tf_raw_ops_digamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Digamma'.")

check_valid('tf.raw_ops.Digamma', generated_inputs['tf.raw_ops.Digamma'], lib="tf", suffix=0)
