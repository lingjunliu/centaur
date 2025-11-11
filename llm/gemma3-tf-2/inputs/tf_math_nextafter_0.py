
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_nextafter_inputs():
    list_of_inputs = []

    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.1, 2.2, 2.9], dtype=np.float32)
    name = "example1"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    x2 = np.array([-0.9, -2.1, -2.9], dtype=np.float64)
    name = "example2"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([1.0], dtype=np.float32)
    x2 = np.array([1.000001], dtype=np.float32)
    name = "example3"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float64)
    name = "example4"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([np.finfo(np.float32).smallest_normal], dtype=np.float32)
    x2 = np.array([np.finfo(np.float32).smallest_normal * 2], dtype=np.float32)
    name = "example5"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([np.finfo(np.float64).smallest_normal], dtype=np.float64)
    x2 = np.array([np.finfo(np.float64).smallest_normal * 2], dtype=np.float64)
    name = "example6"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([0.0], dtype=np.float32)
    x2 = np.array([1e-38], dtype=np.float32)
    name = "example7"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([1e38], dtype=np.float32)
    x2 = np.array([np.finfo(np.float32).max], dtype=np.float32)
    name = "example8"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([-1e38], dtype=np.float64)
    x2 = np.array([-np.finfo(np.float64).max], dtype=np.float64)
    name = "example9"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.array([0.9, 1.9, 2.9], dtype=np.float64)
    name = "example10"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.nextafter"] = tf_math_nextafter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.nextafter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.nextafter', generated_inputs['tf.math.nextafter'], lib="tf", suffix=0)
