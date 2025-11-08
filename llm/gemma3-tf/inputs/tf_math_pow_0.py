
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_pow_inputs():
    list_of_inputs = []

    x = np.array([[2, 2], [3, 3]], dtype=np.float32)
    y = np.array([[8, 16], [2, 3]], dtype=np.float32)
    name = "pow_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    name = "pow_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, -2, -3], dtype=np.float64)
    y = np.array([2, 3, 4], dtype=np.float64)
    name = "pow_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float16)
    name = "pow_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int64)
    name = "pow_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2], dtype=np.int32)
    y = np.array([0, 1, 2], dtype=np.int32)
    name = "pow_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    y = np.array([-1, -2, -3], dtype=np.float32)
    name = "pow_example_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2], dtype=np.int64)
    y = np.array([100, 200], dtype=np.int64)
    name = "pow_example_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.pow', generated_inputs['tf.math.pow'], lib="tf", suffix=0)
