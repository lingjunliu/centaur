
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_sobel_edges_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    image1 = tf.constant(np.random.rand(1, 3, 3, 1), dtype=tf.float32).numpy()
    input_dict1 = {"image": image1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multiple channels
    image2 = tf.constant(np.random.rand(1, 5, 5, 3), dtype=tf.float32).numpy()
    input_dict2 = {"image": image2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multiple batches
    image3 = tf.constant(np.random.rand(4, 7, 7, 1), dtype=tf.float32).numpy()
    input_dict3 = {"image": image3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger image size
    image4 = tf.constant(np.random.rand(1, 28, 28, 1), dtype=tf.float32).numpy()
    input_dict4 = {"image": image4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float64 data type
    image5 = tf.constant(np.random.rand(1, 3, 3, 1), dtype=tf.float64).numpy()
    input_dict5 = {"image": image5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different number of channels
    image6 = tf.constant(np.random.rand(1, 4, 4, 4), dtype=tf.float32).numpy()
    input_dict6 = {"image": image6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Small image
    image7 = tf.constant(np.random.rand(1, 2, 2, 1), dtype=tf.float32).numpy()
    input_dict7 = {"image": image7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Batch size of 2
    image8 = tf.constant(np.random.rand(2, 6, 6, 1), dtype=tf.float32).numpy()
    input_dict8 = {"image": image8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Multiple batches and channels
    image9 = tf.constant(np.random.rand(2, 5, 5, 3), dtype=tf.float32).numpy()
    input_dict9 = {"image": image9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Larger Image and Multiple Channels
    image10 = tf.constant(np.random.rand(1, 64, 64, 3), dtype=tf.float32).numpy()
    input_dict10 = {"image": image10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.sobel_edges"] = tf_image_sobel_edges_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.sobel_edges' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.sobel_edges'.")

check_valid('tf.image.sobel_edges', generated_inputs['tf.image.sobel_edges'], lib="tf", suffix=0)
