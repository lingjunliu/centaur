
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_tan_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")], dtype=np.float32)
    input_dict = {'name': 'tan_1', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], dtype=np.float64)
    input_dict = {'name': 'tan_2', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, 0, 1], [-2, -3, -4]], dtype=np.float32)
    input_dict = {'name': 'tan_3', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict = {'name': 'tan_4', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.5, 2.7], [3.1, -4.8]], dtype=np.float64)
    input_dict = {'name': 'tan_5', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float16)
    input_dict = {'name': 'tan_6', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {'name': 'tan_7', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0, np.pi/2], [np.pi, 3*np.pi/2]]], dtype=np.float32)
    input_dict = {'name': 'tan_8', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-np.pi, -np.pi/2, -np.pi/4], dtype=np.float64)
    input_dict = {'name': 'tan_9', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.complex128)
    input_dict = {'name': 'tan_10', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Tan"] = tf_raw_ops_tan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Tan'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Tan', generated_inputs['tf.raw_ops.Tan'], lib="tf", suffix=0)
