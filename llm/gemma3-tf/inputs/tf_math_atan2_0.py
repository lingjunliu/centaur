
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_atan2_inputs():
    list_of_inputs = []

    y = np.array([1.0, -1.0])
    x = np.array([1.0, 1.0])
    name = "atan2_example_1"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([0.0, 0.0])
    x = np.array([1.0, -1.0])
    name = "atan2_example_2"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([-1.0, 1.0])
    x = np.array([-1.0, -1.0])
    name = "atan2_example_3"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, 2.0, 3.0])
    x = np.array([4.0, 5.0, 6.0])
    name = "atan2_example_4"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1.0, -1.0], [2.0, -2.0]])
    x = np.array([[1.0, 1.0], [2.0, 2.0]])
    name = "atan2_example_5"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    x = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    name = "atan2_example_6"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    x = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    name = "atan2_example_7"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -1.0, 0.0], dtype=np.float16)
    x = np.array([1.0, 1.0, -1.0], dtype=np.float16)
    name = "atan2_example_8"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[[1.0, -1.0]], [[2.0, -2.0]]])
    x = np.array([[[1.0, 1.0]], [[2.0, 2.0]]])
    name = "atan2_example_9"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.atan2"] = tf_math_atan2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.atan2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.atan2', generated_inputs['tf.math.atan2'], lib="tf", suffix=0)
