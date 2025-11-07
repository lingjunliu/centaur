
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_log1p_inputs():
    list_of_inputs = []

    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_0"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.5, -1.0, -2.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "log1p_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.0, 0.5], [1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-5, 1e-2, 1e-1], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e5, 1e6, 1e7], dtype=np.float64)
    input_dict = {"x": x, "name": "log1p_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0 + 1j, 1 + 1j, 2 + 1j], dtype=np.complex64)
    input_dict = {"x": x, "name": "log1p_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 0.5, 1.0], dtype=np.float16)
    input_dict = {"x": x, "name": "log1p_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.complex128)
    input_dict = {"x": x, "name": "log1p_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log1p'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.log1p', generated_inputs['tf.math.log1p'], lib="tf", suffix=0)
