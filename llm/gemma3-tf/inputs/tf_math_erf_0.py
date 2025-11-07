
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_erf_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.5, 0.0, 1.5]], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.5], [1.0]], [[1.5], [2.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    input_dict = {"x": x, "name": "erf_test_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e5, 1e6, 1e7, 1e8, 1e9], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[np.nan, 2.0], [3.0, np.inf]], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-np.inf, -2.0], [3.0, np.nan]], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.erf"] = tf_math_erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.erf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.erf', generated_inputs['tf.math.erf'], lib="tf", suffix=0)
