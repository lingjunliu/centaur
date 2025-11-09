
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []

    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([], dtype=np.int32)
    shape_y = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([4, 1], dtype=np.int32)
    shape_y = np.array([2, 4, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([0, 3], dtype=np.int32)
    shape_y = np.array([0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 2, 1, 4, 1], dtype=np.int32)
    shape_y = np.array([3, 1, 5, 1, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([6, 1, 8], dtype=np.int32)
    shape_y = np.array([1, 8], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([], dtype=np.int32)
    shape_y = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 1, 1], dtype=np.int32)
    shape_y = np.array([9, 8, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1], dtype=np.int32)
    shape_y = np.array([2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1024, 1, 64], dtype=np.int32)
    shape_y = np.array([1, 32, 64], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([2, 0, 3], dtype=np.int32)
    shape_y = np.array([1, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([3], dtype=np.int32)
    shape_y = np.array([1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 5, 1, 1], dtype=np.int32)
    shape_y = np.array([7, 1, 9, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([2, 3, 4], dtype=np.int32)
    shape_y = np.array([1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 0, 1], dtype=np.int32)
    shape_y = np.array([5, 0, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([8, 1, 1], dtype=np.int32)
    shape_y = np.array([1, 1, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape_1"] = tf_broadcast_dynamic_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.broadcast_dynamic_shape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.broadcast_dynamic_shape_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.broadcast_dynamic_shape', generated_inputs['tf.broadcast_dynamic_shape_1'], lib="tf", suffix=1)
