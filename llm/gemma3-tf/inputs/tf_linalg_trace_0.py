
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_trace_inputs():
    list_of_inputs = []

    x = np.array([[1, 2], [3, 4]])
    input_dict = {'x': x, 'name': 'trace_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    input_dict = {'x': x, 'name': 'trace_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]])
    input_dict = {'x': x, 'name': 'trace_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {'x': x, 'name': 'trace_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    input_dict = {'x': x, 'name': 'trace_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, -2], [-3, 4]])
    input_dict = {'x': x, 'name': 'trace_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    input_dict = {'x': x, 'name': 'trace_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]])
    input_dict = {'x': x, 'name': 'trace_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    input_dict = {'x': x, 'name': 'trace_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {'x': x, 'name': 'trace_10'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.trace"] = tf_linalg_trace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.trace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.trace'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.trace', generated_inputs['tf.linalg.trace'], lib="tf", suffix=0)
