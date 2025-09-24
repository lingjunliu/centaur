
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_HSVToRGB_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D float32 tensor
    images = np.array([[[0.5, 0.6, 0.7]]], dtype=np.float32)
    input_dict = {"images": images, "name": "hsv_to_rgb_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D half tensor
    images = np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]], dtype=np.float16)
    input_dict = {"images": images, "name": "hsv_to_rgb_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 tensor (Replaced bfloat16 with float32)
    images = np.array([[0.8, 0.9, 0.0]], dtype=np.float32)
    input_dict = {"images": images, "name": "hsv_to_rgb_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger 3D float64 tensor
    images = np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 0.2]], [[0.4, 0.6, 0.8], [1.0, 0.2, 0.4]]], dtype=np.float64)
    input_dict = {"images": images, "name": "hsv_to_rgb_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 tensor with values outside [0, 1]
    images = np.array([[[1.2, -0.3, 0.5]]], dtype=np.float32)
    input_dict = {"images": images, "name": "hsv_to_rgb_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 tensor
    images = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": "hsv_to_rgb_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 tensor
    images = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": "hsv_to_rgb_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 tensor with different hsv values
    images = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
    input_dict = {"images": images, "name": "hsv_to_rgb_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32 tensor
    images = np.random.rand(2, 2, 2, 2, 3).astype(np.float32)
    input_dict = {"images": images, "name": "hsv_to_rgb_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float64 tensor
    images = np.random.rand(2, 3, 4, 3).astype(np.float64)
    input_dict = {"images": images, "name": "hsv_to_rgb_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.HSVToRGB"] = tf_raw_ops_HSVToRGB_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.HSVToRGB' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.HSVToRGB'.")

check_valid('tf.raw_ops.HSVToRGB', generated_inputs['tf.raw_ops.HSVToRGB'], lib="tf", suffix=0)
