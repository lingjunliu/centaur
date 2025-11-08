
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_logical_not_inputs():
    list_of_inputs = []

    x = np.array([True, False])
    input_dict = {"x": tf.constant(x), "name": "test_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([False, True, False])
    input_dict = {"x": tf.constant(x), "name": "test_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[True, False], [False, True]])
    input_dict = {"x": tf.constant(x), "name": "test_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    input_dict = {"x": tf.constant(x), "name": "test_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([True])
    input_dict = {"x": tf.constant(x), "name": "test_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([False])
    input_dict = {"x": tf.constant(x), "name": "test_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[True]])
    input_dict = {"x": tf.constant(x), "name": "test_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[False]])
    input_dict = {"x": tf.constant(x), "name": "test_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([True, True, True])
    input_dict = {"x": tf.constant(x), "name": "test_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([False, False, False])
    input_dict = {"x": tf.constant(x), "name": "test_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[True, True], [False, False]])
    input_dict = {"x": tf.constant(x), "name": "test_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.logical_not"] = tf_math_logical_not_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.logical_not'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.logical_not', generated_inputs['tf.math.logical_not'], lib="tf", suffix=0)
