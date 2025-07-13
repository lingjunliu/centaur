
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_adjust_hue_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    images = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    delta = np.array(0.5, dtype=np.float32)
    name = None
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple images, small delta
    images = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    delta = np.array(0.1, dtype=np.float32)
    name = "adjust_hue_op_2"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger delta value
    images = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    delta = np.array(1.5, dtype=np.float32)
    name = "adjust_hue_op_3"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative delta value
    images = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    delta = np.array(-0.5, dtype=np.float32)
    name = "adjust_hue_op_4"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D Input
    images = np.random.rand(2, 3, 4, 3).astype(np.float32)
    delta = np.array(0.2, dtype=np.float32)
    name = "adjust_hue_op_5"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Half type
    images = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float16)
    delta = np.array(0.3, dtype=np.float32)
    name = "adjust_hue_op_6"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D Input with different values
    images = np.array([[[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]], [[0.5, 0.7, 0.9], [0.1, 0.2, 0.3]]], dtype=np.float32)
    delta = np.array(0.7, dtype=np.float32)
    name = "adjust_hue_op_7"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Delta > 1
    images = np.array([[[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]]], dtype=np.float32)
    delta = np.array(2.0, dtype=np.float32)
    name = "adjust_hue_op_8"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Delta < -1
    images = np.array([[[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]]], dtype=np.float32)
    delta = np.array(-2.0, dtype=np.float32)
    name = "adjust_hue_op_9"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: larger image size, positive delta
    images = np.random.rand(50, 50, 3).astype(np.float32)
    delta = np.array(0.6, dtype=np.float32)
    name = "adjust_hue_op_10"
    input_dict = {"images": images, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AdjustHue"] = tf_raw_ops_adjust_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AdjustHue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AdjustHue'.")

check_valid('tf.raw_ops.AdjustHue', generated_inputs['tf.raw_ops.AdjustHue'], lib="tf", suffix=0)
