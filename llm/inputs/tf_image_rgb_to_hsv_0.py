
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []

    # Input 1: Basic 3x3 image
    image1 = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict = {"images": image1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2 images
    image2 = np.random.rand(2, 4, 4, 3).astype(np.float32)
    input_dict = {"images": image2, "name": "batch_images"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single pixel image
    image3 = np.random.rand(1, 1, 3).astype(np.float32)
    input_dict = {"images": image3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger image 10x10
    image4 = np.random.rand(10, 10, 3).astype(np.float32)
    input_dict = {"images": image4, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Different name
    image5 = np.random.rand(5, 5, 3).astype(np.float32)
    input_dict = {"images": image5, "name": "different_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Using half type
    image6 = np.random.rand(3, 3, 3).astype(np.float16)
    input_dict = {"images": image6, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Using bfloat16 type
    image7 = np.random.rand(3, 3, 3).astype(np.float16)
    input_dict = {"images": image7, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using float64 type
    image8 = np.random.rand(3, 3, 3).astype(np.float64)
    input_dict = {"images": image8, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D image
    image9 = np.random.rand(3).astype(np.float32)
    input_dict = {"images": image9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D image
    image10 = np.random.rand(2, 3, 4, 3).astype(np.float32)
    input_dict = {"images": image10, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.rgb_to_hsv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_hsv'.")

check_valid('tf.image.rgb_to_hsv', generated_inputs['tf.image.rgb_to_hsv'], lib="tf", suffix=0)
