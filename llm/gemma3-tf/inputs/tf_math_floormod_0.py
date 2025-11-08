
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_floormod_inputs():
    list_of_inputs = []

    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10, -20, -30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.5, 20.5, 30.5], dtype=np.float32)
    y = np.array([3.2, 7.1, 2.5], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    y = np.array([3, 7], dtype=np.int64)
    input_dict = {'x': x, 'y': y, 'name': "floormod_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.uint8)
    y = np.array([3, 7, 2], dtype=np.uint8)
    input_dict = {'x': x, 'y': y, 'name': "floormod_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.float64)
    y = np.array([3.0, 7.0, 2.0], dtype=np.float64)
    input_dict = {'x': x, 'y': y, 'name': "floormod_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32)
    y = np.array([3, 7], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, -20, 30], dtype=np.int16)
    y = np.array([3, 7, 2], dtype=np.int16)
    input_dict = {'x': x, 'y': y, 'name': "floormod_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.float16)
    y = np.array([3, 7, 2], dtype=np.float16)
    input_dict = {'x': x, 'y': y, 'name': "floormod_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.floormod"] = tf_math_floormod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.floormod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floormod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.floormod', generated_inputs['tf.math.floormod'], lib="tf", suffix=0)
