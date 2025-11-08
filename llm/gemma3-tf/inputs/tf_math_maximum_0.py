
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_maximum_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([0.0, 2.0, 1.0, 5.0], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[0, 1], [2, 3]], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 3.0], dtype=np.float64)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([0, 1, 2], dtype=np.uint8)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-5, 0, 5], dtype=np.int16)
    y = np.array([-2, 1, 3], dtype=np.int16)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float16)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    y = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_10'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.maximum"] = tf_math_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.maximum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.maximum', generated_inputs['tf.math.maximum'], lib="tf", suffix=0)
