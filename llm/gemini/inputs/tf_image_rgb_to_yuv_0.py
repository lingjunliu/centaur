
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_yuv_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D image
    images = np.random.rand(10, 10, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D image (batch of images)
    images = np.random.rand(5, 20, 20, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small image
    images = np.random.rand(2, 2, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large image
    images = np.random.rand(100, 100, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Image with values close to 0
    images = np.random.rand(10, 10, 3).astype(np.float32) * 0.1
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Image with values close to 1
    images = np.random.rand(10, 10, 3).astype(np.float32) * 0.1 + 0.9
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor
    images = np.random.rand(2, 5, 5, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D image with different dimensions
    images = np.random.rand(32, 64, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another batch of images
    images = np.random.rand(4, 16, 16, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 2D image
    images = np.random.rand(64, 32, 3).astype(np.float32)
    input_dict = {"images": tf.convert_to_tensor(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.rgb_to_yuv"] = tf_image_rgb_to_yuv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.rgb_to_yuv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_yuv'.")

check_valid('tf.image.rgb_to_yuv', generated_inputs['tf.image.rgb_to_yuv'], lib="tf", suffix=0)
