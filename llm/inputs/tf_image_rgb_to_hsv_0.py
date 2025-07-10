
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D image
    image1 = np.random.rand(5, 5, 3).astype(np.float32)
    input_dict1 = {"images": tf.convert_to_tensor(image1).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D image (batch of images)
    image2 = np.random.rand(2, 10, 10, 3).astype(np.float32)
    input_dict2 = {"images": tf.convert_to_tensor(image2).numpy(), "name": "batch_of_images"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D image (single pixel)
    image3 = np.random.rand(3).astype(np.float32)
    input_dict3 = {"images": tf.convert_to_tensor(image3).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger image
    image4 = np.random.rand(100, 100, 3).astype(np.float32)
    input_dict4 = {"images": tf.convert_to_tensor(image4).numpy(), "name": "large_image"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Using half type
    image5 = np.random.rand(5, 5, 3).astype(np.float16)
    input_dict5 = {"images": tf.convert_to_tensor(image5).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Using bfloat16 type
    image6 = np.random.rand(5, 5, 3).astype(tf.bfloat16.as_numpy_dtype)
    input_dict6 = {"images": tf.convert_to_tensor(image6).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Using float64 type
    image7 = np.random.rand(5, 5, 3).astype(np.float64)
    input_dict7 = {"images": tf.convert_to_tensor(image7).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

   # Input 8: Image with values close to 0 and 1
    image8 = np.random.uniform(low=0.0, high=1.0, size=(5, 5, 3)).astype(np.float32)
    input_dict8 = {"images": tf.convert_to_tensor(image8).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 4D image
    image9 = np.random.rand(1, 2, 3, 3).astype(np.float32)
    input_dict9 = {"images": tf.convert_to_tensor(image9).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Image with a specific name
    image10 = np.random.rand(3, 4, 3).astype(np.float32)
    input_dict10 = {"images": tf.convert_to_tensor(image10).numpy(), "name": "my_image"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
