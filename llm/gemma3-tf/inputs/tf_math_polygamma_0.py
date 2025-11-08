
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_polygamma_inputs():
    list_of_inputs = []

    a = np.float32(1.0)
    x = np.float32(2.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float64(2.0)
    x = np.float64(3.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(0.0)
    x = np.float32(1.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float64(3.0)
    x = np.float64(-1.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(1.0)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0], dtype=np.float64)
    x = np.float64(4.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(2.0)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.float64(5.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(5.0)
    x = np.float32(0.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float64(1.5)
    x = np.float64(2.5)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.polygamma"] = tf_math_polygamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.polygamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.polygamma', generated_inputs['tf.math.polygamma'], lib="tf", suffix=0)
