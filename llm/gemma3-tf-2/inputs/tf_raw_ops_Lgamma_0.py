
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_lgamma_inputs():
    list_of_inputs = []

    x = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_1', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -0.5, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {'name': 'lgamma_test_2', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_3', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {'name': 'lgamma_test_4', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    input_dict = {'name': 'lgamma_test_5', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_6', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_7', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5.5, 6.5, 7.5], dtype=np.float64)
    input_dict = {'name': 'lgamma_test_8', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_9', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_10', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Lgamma"] = tf_raw_ops_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Lgamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Lgamma', generated_inputs['tf.raw_ops.Lgamma'], lib="tf", suffix=0)
