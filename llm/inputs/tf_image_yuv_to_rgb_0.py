
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_yuv_to_rgb_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D image
    images = np.random.rand(32, 32, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.yuv_to_rgb"] = tf_image_yuv_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.yuv_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.yuv_to_rgb'.")

check_valid('tf.image.yuv_to_rgb', generated_inputs['tf.image.yuv_to_rgb'], lib="tf", suffix=0)
