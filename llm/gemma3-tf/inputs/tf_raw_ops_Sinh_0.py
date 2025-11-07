
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Sinh_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    input_dict = {'x': x, 'name': 'sinh_test_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.5, 0, 2.7, -3.1], dtype=np.float64)
    input_dict = {'x': x, 'name': 'sinh_test_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, 2-2j, -1-1j], dtype=np.complex64)
    input_dict = {'x': x, 'name': 'sinh_test_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'sinh_test_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.half)
    input_dict = {'x': x, 'name': 'sinh_test_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
    input_dict = {'x': x, 'name': 'sinh_test_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'sinh_test_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-100, 0, 100], dtype=np.float64)
    input_dict = {'x': x, 'name': 'sinh_test_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sinh"] = tf_raw_ops_Sinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sinh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Sinh', generated_inputs['tf.raw_ops.Sinh'], lib="tf", suffix=0)
