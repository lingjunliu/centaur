
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_digamma_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict = {"x": x, "name": "digamma_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, scalar value
    x = tf.constant(1.0, dtype=tf.bfloat16).numpy()
    input_dict = {"x": x.astype(np.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    input_dict = {"x": x, "name": "another_digamma"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, scalar value
    x = np.array(0.1, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 1D array with negative values. Should still work as it computes Digamma for absolute values.
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {"x": x, "name": "negative_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16, 2D array
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.bfloat16).numpy()
    input_dict = {"x": x.astype(np.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, scalar
    x = np.array(2.5, dtype=np.float16)
    input_dict = {"x": x, "name": "scalar_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, multi-dimensional array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, small values
    x = np.array([0.001, 0.002, 0.003], dtype=np.float64)
    input_dict = {"x": x, "name": None}
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
