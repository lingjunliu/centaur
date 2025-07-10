
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_lbeta_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "lbeta_1"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    name = "lbeta_2"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "lbeta_3"
    input_dict = {"x": tf.constant([x]), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    name = "lbeta_4"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    name = "lbeta_5"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    name = "lbeta_6"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[3.0, 2.0], [5.0, 4.0]], dtype=np.float32)
    name = "lbeta_7"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float32)
    name = "lbeta_8"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    name = "lbeta_9"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    name = "lbeta_10"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp = tf_math_lbeta_inputs()
for i in range(len(temp)):
    temp[i]['x'] = temp[i]['x'].numpy()
generated_inputs["tf.math.lbeta"] = temp

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.lbeta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.lbeta'.")

check_valid('tf.math.lbeta', generated_inputs['tf.math.lbeta'], lib="tf", suffix=0)
