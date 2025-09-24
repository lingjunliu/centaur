
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_adjust_saturation_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D float32 image with positive scale
    images = np.array([[[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]], dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 3D half image with positive scale
    images = np.array([[[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]], dtype=np.float16)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D float32 image with positive scale
    images = np.random.rand(2, 3, 4, 3).astype(np.float32)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 image with negative scale
    images = np.array([[[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]]], dtype=np.float32)
    scale = np.array(-0.3, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger 3D float32 image with scale 0
    images = np.random.rand(10, 10, 3).astype(np.float32)
    scale = np.array(0.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D half image with scale greater than 1
    images = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float16)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D half image
    images = np.random.rand(5, 5, 5, 3).astype(np.float16)
    scale = np.array(0.7, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 image with scale = 1
    images = np.array([[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]], dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 with 3 values near 0
    images = np.array([[[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]]], dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different sized image
    images = np.random.rand(2, 5, 3).astype(np.float32)
    scale = np.array(0.8, dtype=np.float32)
    input_dict = {"images": images, "scale": scale, "name": "adjust_sat_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AdjustSaturation"] = tf_raw_ops_adjust_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AdjustSaturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AdjustSaturation'.")

check_valid('tf.raw_ops.AdjustSaturation', generated_inputs['tf.raw_ops.AdjustSaturation'], lib="tf", suffix=0)
