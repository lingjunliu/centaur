
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_round_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values, decimals=0
    a = tf.constant(np.array([1.2, 2.5, 3.7, 4.0]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values, decimals=0
    a = tf.constant(np.array([-1.2, -2.5, -3.7, -4.0]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Decimals=1
    a = tf.constant(np.array([1.23, 2.56, 3.78, 4.01]))
    decimals = 1
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decimals=2
    a = tf.constant(np.array([1.234, 2.567, 3.789, 4.012]))
    decimals = 2
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, decimals=0
    a = tf.constant(np.array([[1.2, 2.5], [3.7, 4.0]]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, decimals=1
    a = tf.constant(np.array([[1.23, 2.56], [3.78, 4.01]]))
    decimals = 1
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, decimals=0
    a = tf.constant(np.array([[[1.2, 2.5], [3.7, 4.0]], [[5.1, 6.8], [7.3, 8.6]]]))
    decimals = 0
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large decimals (greater than precision), decimals=5
    a = tf.constant(np.array([1.23456789, 2.56789012]))
    decimals = 5
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative decimals, decimals=-1
    a = tf.constant(np.array([12.3, 25.6, 37.8, 40.1]))
    decimals = -1
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative decimals and negative numbers, decimals=-2
    a = tf.constant(np.array([-123.4, -256.7, -378.9, -401.2]))
    decimals = -2
    input_dict = {"a": a, "decimals": decimals}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.round"] = tf_experimental_numpy_round_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.round'.")

check_valid('tf.experimental.numpy.round', generated_inputs['tf.experimental.numpy.round'], lib="tf", suffix=0)
