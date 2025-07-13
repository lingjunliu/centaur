
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_HSVToRGB_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D input with float32
    images = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    name = None
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D input with float64
    images = np.random.rand(2, 3, 3).astype(np.float64)
    name = "hsv_to_rgb_1"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D input with half
    images = np.array([0.7, 0.8, 0.9], dtype=np.float16)
    name = "hsv_to_rgb_2"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D input with bfloat16
    images = np.random.rand(1, 2, 3, 3).astype(tf.bfloat16.as_numpy_dtype)
    name = "hsv_to_rgb_3"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All zeros with float32
    images = np.zeros((2, 3, 3), dtype=np.float32)
    name = "hsv_to_rgb_4"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All ones with float64
    images = np.ones((1, 1, 3), dtype=np.float64)
    name = "hsv_to_rgb_5"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different values with half
    images = np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.2]], dtype=np.float16)
    name = "hsv_to_rgb_6"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D input with bfloat16
    images = np.random.rand(1, 2, 1, 3, 3).astype(tf.bfloat16.as_numpy_dtype)
    name = "hsv_to_rgb_7"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Values close to 1 with float32
    images = np.array([[0.99, 0.98, 0.97], [0.96, 0.95, 0.94]], dtype=np.float32)
    name = "hsv_to_rgb_8"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 3D array, float 32
    images = np.random.rand(4, 5, 3).astype(np.float32)
    name = "hsv_to_rgb_9"
    input_dict = {"images": images, "name": name}
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
