
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_pow_inputs():
    list_of_inputs = []

    x = np.array([[2, 2], [3, 3]], dtype=np.float32)
    y = np.array([[8, 16], [2, 3]], dtype=np.float32)
    input_dict = {'name': 'pow_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {'name': 'pow_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, 2], dtype=np.float64)
    y = np.array([3, 2], dtype=np.float64)
    input_dict = {'name': 'pow_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    y = np.array([[2, 3], [4, 5]], dtype=np.complex64)
    input_dict = {'name': 'pow_4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.5, 1.5], dtype=np.float32)
    y = np.array([2, 0.5], dtype=np.float32)
    input_dict = {'name': 'pow_5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10], dtype=np.int8)
    y = np.array([2], dtype=np.int8)
    input_dict = {'name': 'pow_6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2.0], dtype=np.float64)
    y = np.array([-1.0], dtype=np.float64)
    input_dict = {'name': 'pow_7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([1, 1, 1], dtype=np.int16)
    input_dict = {'name': 'pow_8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0], dtype=np.float32)
    y = np.array([0], dtype=np.float32)
    input_dict = {'name': 'pow_9', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Pow', generated_inputs['tf.raw_ops.Pow'], lib="tf", suffix=0)
