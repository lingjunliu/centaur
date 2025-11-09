
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []

    shape_x = [np.int32(1), np.int32(2), np.int32(3)]
    shape_y = [np.int32(5), np.int32(1), np.int32(3)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(3), np.int64(1)]
    shape_y = [np.int64(2), np.int64(3), np.int64(4)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(1)]
    shape_y = [np.int32(7)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = []
    shape_y = [np.int64(4), np.int64(5)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(4), np.int64(1), np.int64(1)]
    shape_y = [np.int64(1), np.int64(5), np.int64(6)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(2), np.int32(0), np.int32(3)]
    shape_y = [np.int32(1), np.int32(0), np.int32(1)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(6), np.int64(7), np.int64(1)]
    shape_y = [np.int64(1), np.int64(7), np.int64(8)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [1, 1, 1, 1]
    shape_y = [np.int32(9)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(32), np.int32(1), np.int32(224), np.int32(224)]
    shape_y = [np.int32(1), np.int32(3), np.int32(1), np.int32(1)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(1), np.int64(0)]
    shape_y = [np.int64(5), np.int64(0)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = []
    shape_y = []
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(2), np.int32(3), np.int32(4)]
    shape_y = [np.int32(1), np.int32(4)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(10), np.int64(11)]
    shape_y = [np.int64(1), np.int64(1)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(1), np.int32(2), np.int32(1), np.int32(3)]
    shape_y = [np.int32(3)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(7), np.int32(1), np.int32(5)]
    shape_y = [np.int32(1), np.int32(8), np.int32(5)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(1), np.int64(1), np.int64(0)]
    shape_y = [np.int64(3), np.int64(4), np.int64(0)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape_3"] = tf_broadcast_dynamic_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.broadcast_dynamic_shape_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.broadcast_dynamic_shape_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.broadcast_dynamic_shape', generated_inputs['tf.broadcast_dynamic_shape_3'], lib="tf", suffix=3)
