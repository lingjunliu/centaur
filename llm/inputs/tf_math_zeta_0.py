
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_zeta_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.5], dtype=np.float32)
    q = np.array([0.5], dtype=np.float32)
    name = "zeta_1"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([2.0], dtype=np.float64)
    q = np.array([1.0], dtype=np.float64)
    name = "zeta_2"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    q = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    name = "zeta_3"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[2.0, 2.5], [3.0, 3.5]], dtype=np.float64)
    q = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64)
    name = "zeta_4"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    name = "zeta_5"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    x = np.array([-1.0], dtype=np.float32)
    q = np.array([1.0], dtype=np.float32)
    name = "zeta_6"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    q = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    name = "zeta_7"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([0.9999], dtype=np.float32)
    q = np.array([1.0], dtype=np.float32)
    name = "zeta_8"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1.0001], dtype=np.float64)
    q = np.array([1.0], dtype=np.float64)
    name = "zeta_9"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1.5, 2.0], [2.5, 3.0]], [[3.5, 4.0], [4.5, 5.0]]], dtype=np.float32)
    q = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float32)
    name = "zeta_10"
    input_dict = {"x": tf.constant(x), "q": tf.constant(q), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_list = tf_math_zeta_inputs()
for input_dict in temp_list:
    input_dict["x"] = input_dict["x"].numpy()
    input_dict["q"] = input_dict["q"].numpy()
generated_inputs["tf.math.zeta"] = temp_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.zeta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.zeta'.")

check_valid('tf.math.zeta', generated_inputs['tf.math.zeta'], lib="tf", suffix=0)
