
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_gelu_inputs():
    list_of_inputs = []

    # Input 1
    features = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32).numpy()
    approximate = False
    name = None
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    features = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32).numpy()
    approximate = True
    name = "gelu_approx"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    features = tf.constant(np.random.randn(2, 3), dtype=np.float32).numpy()
    approximate = False
    name = "gelu_2d"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    features = tf.constant(np.random.randn(1, 4, 5), dtype=np.float32).numpy()
    approximate = True
    name = "gelu_3d_approx"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    features = tf.constant(np.array([[-100.0, -0.001], [0.001, 100.0]]), dtype=np.float32).numpy()
    approximate = False
    name = "gelu_extreme_values"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    features = tf.constant(np.zeros((2, 2)), dtype=np.float32).numpy()
    approximate = True
    name = "gelu_zeros"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    features = tf.constant(np.ones((3, 3)), dtype=np.float32).numpy()
    approximate = False
    name = "gelu_ones"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    features = tf.constant(np.random.randn(10), dtype=np.float32).numpy()
    approximate = True
    name = "gelu_random_approx"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    features = tf.constant(np.random.randn(2, 2, 2, 2), dtype=np.float32).numpy()
    approximate = False
    name = "gelu_4d"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    features = tf.constant([-1e-8, 1e-8], dtype=tf.float32).numpy()
    approximate = True
    name = "gelu_small_values"
    input_dict = {"features": features, "approximate": approximate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.gelu"] = tf_nn_gelu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.gelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.gelu'.")

check_valid('tf.nn.gelu', generated_inputs['tf.nn.gelu'], lib="tf", suffix=0)
