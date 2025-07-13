
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RGBToHSV_inputs():
    list_of_inputs = []

    # Input 1: Basic 3x3 image, float32
    images = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1x1 image, half
    images = np.random.rand(1, 1, 3).astype(np.float16)
    input_dict = {"images": images, "name": "small_image"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of images, float64
    images = np.random.rand(5, 10, 10, 3).astype(np.float64)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array representing a single pixel, float32
    images = np.random.rand(3).astype(np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger image, float32
    images = np.random.rand(100, 100, 3).astype(np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Image with a different name
    images = np.random.rand(5, 5, 3).astype(np.float32)
    input_dict = {"images": images, "name": "different_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Image with zeros
    images = np.zeros((5, 5, 3)).astype(np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Image with ones
    images = np.ones((5, 5, 3)).astype(np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Image with mixed values
    images = np.array([[[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]],
                       [[0.5, 0.7, 0.9], [0.0, 0.2, 0.4]]]).astype(np.float32)
    input_dict = {"images": images, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RGBToHSV"] = tf_raw_ops_RGBToHSV_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RGBToHSV' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RGBToHSV'.")

check_valid('tf.raw_ops.RGBToHSV', generated_inputs['tf.raw_ops.RGBToHSV'], lib="tf", suffix=0)
