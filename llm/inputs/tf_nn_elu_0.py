
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_elu_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32
    features = tf.constant(1.0, dtype=tf.float32).numpy()
    name = "elu_scalar_float32"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar float64
    features = tf.constant(-1.0, dtype=tf.float64).numpy()
    name = "elu_scalar_float64"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor float32
    features = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32).numpy()
    name = "elu_1d_float32"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor float32
    features = tf.constant([[-1.0, 0.0], [1.0, 2.0]], dtype=tf.float32).numpy()
    name = "elu_2d_float32"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor float32
    features = tf.constant([[[0.0, 1.0], [-1.0, -2.0]], [[2.0, -3.0], [3.0, 4.0]]], dtype=tf.float32).numpy()
    name = "elu_3d_float32"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor float64
    features = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float64).numpy()
    name = "elu_1d_float64"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tensor float64
    features = tf.constant([[-1.0, 0.0], [1.0, 2.0]], dtype=tf.float64).numpy()
    name = "elu_2d_float64"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with only negative values (float32)
    features = tf.constant([-5.0, -3.0, -1.0], dtype=tf.float32).numpy()
    name = "elu_negative_float32"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with only positive values (float32)
    features = tf.constant([5.0, 3.0, 1.0], dtype=tf.float32).numpy()
    name = "elu_positive_float32"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.elu"] = tf_nn_elu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.elu'.")

check_valid('tf.nn.elu', generated_inputs['tf.nn.elu'], lib="tf", suffix=0)
