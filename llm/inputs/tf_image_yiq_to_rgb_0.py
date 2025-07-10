
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_yiq_to_rgb_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    images = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple images
    images = np.array([[[0.2, 0.1, -0.1], [0.8, -0.2, 0.2]], [[0.5, 0.0, 0.0], [0.1, 0.5, -0.5]]], dtype=np.float32)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different valid YIQ values
    images = np.array([[[0.0, -0.5957, -0.5226], [1.0, 0.5957, 0.5226]]], dtype=np.float32)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Images with more channels (should still work)
    images = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float64)
    input_dict = {"images": tf.constant(images, dtype=tf.float32).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single pixel image
    images = np.array([[[0.3, 0.2, 0.1]]], dtype=np.float32)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Grayscale image
    images = np.array([[[0.7, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of grayscale images
    images = np.array([[[0.4, 0.0, 0.0]], [[0.6, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: 4D input
    images = np.random.rand(2, 3, 4, 3).astype(np.float32)
    images[:,:,:,0] = np.clip(images[:,:,:,0], 0, 1)
    images[:,:,:,1] = np.clip(images[:,:,:,1], -0.5957, 0.5957)
    images[:,:,:,2] = np.clip(images[:,:,:,2], -0.5226, 0.5226)

    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small values
    images = np.array([[[0.001, 0.001, 0.001]]], dtype=np.float32)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger image
    images = np.random.rand(10, 10, 3).astype(np.float32)
    images[:,:,0] = np.clip(images[:,:,0], 0, 1)
    images[:,:,1] = np.clip(images[:,:,1], -0.5957, 0.5957)
    images[:,:,2] = np.clip(images[:,:,2], -0.5226, 0.5226)
    input_dict = {"images": tf.constant(images).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.yiq_to_rgb"] = tf_image_yiq_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.yiq_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.yiq_to_rgb'.")

check_valid('tf.image.yiq_to_rgb', generated_inputs['tf.image.yiq_to_rgb'], lib="tf", suffix=0)
