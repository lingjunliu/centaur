
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_silu_inputs():
    list_of_inputs = []

    # Input 1: Basic test with positive values
    features = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Test with negative values
    features = tf.constant(np.array([-1.0, -2.0, -3.0], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with zero values
    features = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with different beta value
    features = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)).numpy()
    beta = tf.constant(0.5, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with different beta value (negative)
    features = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)).numpy()
    beta = tf.constant(-0.5, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Test with a 2D tensor
    features = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Test with a 3D tensor
    features = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Test with large values
    features = tf.constant(np.array([100.0, 200.0, 300.0], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Test with small values
    features = tf.constant(np.array([0.01, 0.02, 0.03], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Test with mixed positive and negative values
    features = tf.constant(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)).numpy()
    beta = tf.constant(1.0, dtype=np.float32).numpy()
    input_dict = {"features": features, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.silu"] = tf_nn_silu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.silu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.silu'.")

check_valid('tf.nn.silu', generated_inputs['tf.nn.silu'], lib="tf", suffix=0)
