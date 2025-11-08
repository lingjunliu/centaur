
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_lgamma_inputs():
    list_of_inputs = []

    x = np.array([0, 0.5, 1, 4.5, -4, -5.6], dtype=np.float32)
    input_dict = {"x": x, "name": "example1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x, "name": "example2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "example3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5.0], dtype=np.float16)
    input_dict = {"x": x, "name": "example4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "example5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "example6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 1.1, 2.1, 3.1], dtype=np.float32)
    input_dict = {"x": x, "name": "example7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.1, -1.1, -2.1, -3.1], dtype=np.float64)
    input_dict = {"x": x, "name": "example8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict = {"x": x, "name": "example9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    input_dict = {"x": x, "name": "example10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.lgamma"] = tf_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.lgamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.lgamma', generated_inputs['tf.math.lgamma'], lib="tf", suffix=0)
